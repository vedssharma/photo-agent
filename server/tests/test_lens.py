"""Lens profile corrections from Lensfun, driven by the photo's own EXIF."""

import io

import cv2
import lensfunpy
import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from photo_agent import color, imaging, lens
from photo_agent import operations as ops
from photo_agent.render import RenderContext, render

CAMERA = ("Canon", "Canon EOS 5D Mark III")
LENS = "EF 24-105mm f/4L IS USM"
W, H = 600, 400


def exif_bytes(lens_model: str = LENS, focal: float = 24.0, aperture: float = 4.0) -> bytes:
    exif = Image.Exif()
    exif[0x010F], exif[0x0110] = CAMERA
    sub = exif.get_ifd(0x8769)
    sub[0xA434] = lens_model
    sub[0x920A] = focal
    sub[0x829D] = aperture
    return exif.tobytes()


SHOT = lens.shot_from_exif(exif_bytes())


def grid() -> np.ndarray:
    """White with dark horizontal and vertical lines every 50 pixels."""
    img = np.ones((H, W, 3), np.float32)
    img[::50] = 0.1
    img[:, ::50] = 0.1
    img[H - 1] = img[:, W - 1] = 0.1
    return img


def as_shot(img: np.ndarray, flags: int) -> np.ndarray:
    """What the lens does to a scene: its distortion or vignetting, applied (Lensfun run in
    reverse)."""
    assert SHOT is not None
    found = lens.find(SHOT)
    assert found is not None
    m = lensfunpy.Modifier(found.lens, found.crop_factor, W, H)
    m.initialize(24.0, 4.0, 1000.0, scale=1.0, pixel_format=np.float32, flags=flags, reverse=True)
    if flags == lensfunpy.ModifyFlags.VIGNETTING:
        linear = np.ascontiguousarray(color.decode_curve(img))
        m.apply_color_modification(linear)
        return np.asarray(color.encode_curve(linear))
    coords = m.apply_geometry_distortion()
    return cv2.remap(img, coords[..., 0], coords[..., 1], cv2.INTER_LINEAR)


def bend(img: np.ndarray) -> float:
    """How far the dark line nearest the top strays from straight, in pixels."""
    dark = img[:, :, 1] < 0.5
    start = int(np.flatnonzero(dark[30 : H // 3, W // 2 + 25])[0]) + 30
    rows = []
    for c in range(75, W - 50, 50):  # Between the vertical lines, clear of the edges.
        near = np.flatnonzero(dark[start - 20 : start + 20, c])
        rows.append(near[np.argmin(np.abs(near - 20))] if near.size else 40)
    return float(np.ptp(rows))


def test_reads_the_lens_from_exif() -> None:
    assert SHOT is not None
    assert (SHOT.make, SHOT.model, SHOT.lens) == (*CAMERA, LENS)
    assert SHOT.focal_length == 24.0 and SHOT.aperture == 4.0
    view = lens.describe(SHOT)
    assert view.profile == "Canon Canon EF 24-105mm f/4L IS USM"
    assert lens.shot_from_exif(b"") is None


def test_straightens_the_lines_the_lens_bends() -> None:
    shot = as_shot(grid(), lensfunpy.ModifyFlags.DISTORTION)
    assert bend(shot) >= 4  # 24 mm barrel distortion bows lines near the edge.
    ctx = RenderContext(shot=SHOT)
    fixed = render(shot, [ops.LensProfile()], ctx)
    assert bend(fixed) <= 1
    # Half the fix bends them half as much.
    half = render(shot, [ops.LensProfile(distortion=50, vignetting=0)], ctx)
    assert bend(fixed) < bend(half) < bend(shot)


def test_brightens_the_corners_the_lens_darkens() -> None:
    flat = np.full((H, W, 3), 0.5, np.float32)
    shot = as_shot(flat, lensfunpy.ModifyFlags.VIGNETTING)
    assert shot[5, 5, 0] < 0.45
    fixed = render(
        shot, [ops.LensProfile(distortion=0, chromatic_aberration=0)], RenderContext(shot=SHOT)
    )
    assert abs(float(fixed[5, 5, 0]) - float(fixed[H // 2, W // 2, 0])) < 0.01


def test_unknown_lenses_are_left_alone() -> None:
    unknown = lens.shot_from_exif(exif_bytes(lens_model="Mystery 50mm f/0.7"))
    img = grid()
    assert lens.find(unknown) is None  # type: ignore[arg-type]
    assert np.array_equal(render(img, [ops.LensProfile()], RenderContext(shot=unknown)), img)
    assert np.array_equal(render(img, [ops.LensProfile()], RenderContext()), img)
    # Naming the lens by hand finds its profile.
    named = ops.LensProfile(lens=LENS)
    assert not np.array_equal(render(img, [named], RenderContext(shot=unknown)), img)


def test_the_profile_applies_before_other_framing() -> None:
    img = as_shot(grid(), lensfunpy.ModifyFlags.DISTORTION)
    crop = ops.Crop(left=0.1, right=0.9)
    ctx = RenderContext(shot=SHOT, source_aspect=W / H)
    first = render(img, [ops.LensProfile(), crop], ctx)
    after = render(img, [crop, ops.LensProfile()], ctx)
    assert np.array_equal(first, after)


def test_lens_profile_end_to_end(client: TestClient) -> None:
    buf = io.BytesIO()
    Image.fromarray(imaging.to_uint8(grid())).save(buf, format="JPEG", exif=exif_bytes())
    res = client.post("/api/documents", files={"file": ("grid.jpg", buf.getvalue())})
    doc = res.json()
    info = client.get(f"/api/documents/{doc['id']}/lens").json()
    assert info["lens"] == LENS and info["profile"].endswith("EF 24-105mm f/4L IS USM")
    assert info["focal_length"] == 24.0

    state = {**doc["state"], "framing": [{"op": "lens_profile"}]}
    res = client.post(
        f"/api/documents/{doc['id']}/edits", json={"label": "Lens profile", "state": state}
    )
    assert res.status_code == 200, res.text
    assert res.json()["state"]["framing"][0]["op"] == "lens_profile"
    assert client.get(f"/api/documents/{doc['id']}/preview").status_code == 200
    specs = {s["op"]: s for s in client.get("/api/operations").json()}
    assert specs["lens_profile"]["framing"]
