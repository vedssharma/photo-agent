"""Phase 7's goal, end to end: an enthusiast's RAW file, developed with its colors and
tones intact, corrected for its lens, soft-proofed, and exported as a 16-bit wide-gamut
TIFF; then a bracket of RAW shots merged to HDR in the library."""

import asyncio
import io

import cv2
import lensfunpy
import numpy as np
from fake_model import FakeModel, text, tool
from fastapi.testclient import TestClient
from PIL import Image
from raw_helpers import make_dng
from test_lens import CAMERA, LENS, bend, grid

from photo_agent import color, imaging, lens
from photo_agent.agent import AgentEvent, AgentService
from photo_agent.layers import EditState
from photo_agent.render import RenderCache, render_state
from photo_agent.routes import get_store
from photo_agent.settings import Settings

W, H = 600, 400
PATCH = (slice(170, 230), slice(270, 330))
"""A swatch of green more saturated than sRGB can show, in the middle of the frame."""
REC2020_GREEN = np.array([0.05, 0.6, 0.05])
"""Linear Rec.2020: well outside sRGB and Display P3."""


def lit() -> np.ndarray:
    """A grid (to show the lens's distortion) with a vivid green swatch, as linear
    Rec.2020 light."""
    linear = color.decode_curve(grid()).astype(np.float64) * 0.6
    linear[PATCH] = REC2020_GREEN
    return linear


def scene() -> np.ndarray:
    """The grid and swatch seen through the lens: bowed by its barrel distortion."""
    linear = lit()
    shot = lens.shot_from_exif(_exif())
    assert shot is not None
    found = lens.find(shot)
    assert found is not None
    m = lensfunpy.Modifier(found.lens, found.crop_factor, W, H)
    flags = lensfunpy.ModifyFlags.DISTORTION
    m.initialize(24.0, 4.0, 1000.0, scale=1.0, pixel_format=np.float32, flags=flags, reverse=True)
    coords = m.apply_geometry_distortion()
    warped = cv2.remap(linear.astype(np.float32), coords[..., 0], coords[..., 1], cv2.INTER_AREA)
    return np.asarray(warped, np.float64)


def _exif() -> bytes:
    exif = Image.Exif()
    exif[0x010F], exif[0x0110] = CAMERA
    sub = exif.get_ifd(0x8769)
    sub[0xA434], sub[0x920A], sub[0x829D] = LENS, 24.0, 4.0
    return exif.tobytes()


def dng(light: np.ndarray, exposure: float = 1.0) -> bytes:
    make, model = CAMERA
    return make_dng(
        light,
        make=make,
        model=model,
        exposure=exposure,
        lens=LENS,
        focal_length=24.0,
        aperture=4.0,
    )


def test_develop_correct_proof_and_export_a_raw_file(
    client: TestClient, settings: Settings
) -> None:
    light = scene()
    res = client.post("/api/documents", files={"file": ("IMG_2401.dng", dng(light))})
    assert res.status_code == 201, res.text
    doc = res.json()
    doc_id = doc["id"]
    # Developed into the wide working space, so the green swatch is not clipped to sRGB.
    assert doc["format"] == "RAW" and doc["color_space"] == color.WIDE

    # The lens is recognized from the file, and Lensfun has a profile for it.
    found = client.get(f"/api/documents/{doc_id}/lens").json()
    assert found["lens"] == LENS and found["focal_length"] == 24.0
    assert found["profile"] == "Canon Canon EF 24-105mm f/4L IS USM"

    model = FakeModel(
        [tool("lens_profile", {}), tool("exposure", {"stops": 0.3})],
        [text("I corrected the lens and brightened the photo a little.")],
    )
    events: list[AgentEvent] = []

    async def emit(event: AgentEvent) -> None:
        events.append(event)

    store = get_store(settings)
    service = AgentService(store, RenderCache(), model)
    edited = asyncio.run(service.run_turn(doc_id, "fix the lens distortion", emit))
    assert [op.op for op in edited.state.framing] == ["lens_profile"]

    loaded = store.image(doc_id)
    ctx = loaded.full_context
    before = render_state(loaded.source.pixels, EditState(), ctx)
    after = render_state(loaded.source.pixels, edited.state, ctx)
    # The lines the lens bowed are straight again.
    assert bend(before) >= 4
    assert bend(after) <= 1.5

    # Soft-proofing: the swatch does not fit sRGB, and the warning paints it.
    gamut = client.get(f"/api/documents/{doc_id}/gamut").json()
    assert gamut["color_space"] == color.WIDE
    assert gamut["outside_srgb"] > 0.005
    assert gamut["outside_display_p3"] <= gamut["outside_srgb"]
    proof = client.get(
        f"/api/documents/{doc_id}/preview", params={"proof": "srgb", "gamut_warning": True}
    )
    assert proof.status_code == 200
    warned = imaging.decode(proof.content).pixels
    ph, pw = warned.shape[:2]
    middle = warned[ph * 210 // H : ph * 220 // H, pw * 295 // W : pw * 305 // W]
    magenta = (middle[..., 0] > 0.8) & (middle[..., 1] < 0.2) & (middle[..., 2] > 0.8)
    assert magenta.mean() > 0.9

    # Exported as a 16-bit Rec.2020 TIFF: the swatch's colors and the smooth tones survive,
    # with the profile and the camera and lens details embedded.
    res = client.post(
        f"/api/documents/{doc_id}/export",
        json={"format": "tiff", "bit_depth": 16, "color_space": "rec2020"},
    )
    assert res.status_code == 200, res.text
    assert res.headers["content-type"] == "image/tiff"
    out = imaging.decode(res.content)
    assert out.space == "rec2020" and out.pixels.shape == (H, W, 3)
    assert len(np.unique(out.pixels[..., 1])) > 4096
    swatch = color.decode_curve(out.pixels[200:210, 295:305]).mean(axis=(0, 1))
    assert swatch[1] > 4 * max(swatch[0], swatch[2])  # Still a vivid wide-gamut green.
    assert not color.out_of_gamut(out.pixels[200:210, 295:305], "rec2020", "rec2020").any()
    assert color.out_of_gamut(out.pixels[200:210, 295:305], "rec2020", "srgb").all()
    kept = Image.Exif()
    kept.load(out.exif)
    assert (kept[0x010F], kept[0x0110]) == CAMERA

    # And the same edit for the web, as an sRGB AVIF.
    web = client.post(
        f"/api/documents/{doc_id}/export", json={"format": "avif", "color_space": "srgb"}
    )
    assert web.status_code == 200
    with Image.open(io.BytesIO(web.content)) as img:
        assert img.format == "AVIF" and img.size == (W, H)


def test_merge_a_raw_bracket_to_hdr_in_the_library(client: TestClient) -> None:
    light = lit()
    light[:60] *= 3.0  # A bright sky the normal exposure clips.
    files = [
        (f"IMG_31{i}.dng", dng(light, exposure)) for i, exposure in enumerate((0.25, 1.0, 4.0))
    ]
    res = client.post("/api/library/photos", files=[("files", f) for f in files])
    assert res.status_code == 201, res.text
    ids = [p["id"] for p in res.json()["imported"]]
    made = client.post("/api/library/merge", json={"ids": ids, "kind": "hdr"})
    assert made.status_code == 201, made.text
    photo = made.json()
    assert photo["format"] == "TIFF" and (photo["width"], photo["height"]) == (W, H)

    # Opened for editing, the merged photo is wide-gamut, with detail in the sky.
    opened = client.get(f"/api/documents/{photo['id']}")
    assert opened.status_code == 200, opened.text
    assert opened.json()["color_space"] == color.WIDE
    merged = imaging.decode(client.get(f"/api/documents/{photo['id']}/original").content)
    normal = imaging.decode(files[1][1])
    sky = (slice(5, 55), slice(40, 160))

    def clipped(pixels: np.ndarray) -> float:
        return float((pixels[sky].min(axis=2) > 0.985).mean())

    # The normal exposure washes most of the sky out to white; the merge keeps it.
    assert clipped(normal.pixels) > 0.5
    assert clipped(merged.pixels) < 0.05
