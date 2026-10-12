"""Color fidelity: wide-gamut working space, high bit depth, and output transforms."""

import io
from pathlib import Path

import cv2
import numpy as np
import pytest
import tifffile
from fastapi.testclient import TestClient
from raw_helpers import make_dng

from photo_agent import color, imaging
from photo_agent import operations as ops
from photo_agent.layers import EditState, Layer
from photo_agent.render import RenderContext, render_state

WIDE_GREEN = np.array([0.05, 0.6, 0.05])
"""A saturated green in linear Rec.2020, well outside what sRGB can show."""


def wide_scene() -> np.ndarray:
    """Linear Rec.2020 light: a gray card on the left, the wide green on the right."""
    scene = np.full((60, 120, 3), 0.18)
    scene[:, 60:] = WIDE_GREEN
    return scene


@pytest.mark.parametrize("space", ["srgb", "display-p3", "rec2020"])
def test_white_is_white_in_every_space(space: color.Space) -> None:
    white = color.to_xyz_matrix(space) @ np.ones(3)
    assert np.allclose(white, [0.9505, 1.0, 1.089], atol=2e-3)


def test_colors_that_fit_round_trip() -> None:
    rng = np.random.default_rng(1)
    srgb = rng.random((32, 32, 3)).astype(np.float32)
    wide = color.convert(srgb, "srgb", "rec2020")
    assert np.abs(color.convert(wide, "rec2020", "srgb") - srgb).max() < 1e-4
    assert color.out_of_gamut(wide, "rec2020", "srgb").sum() == 0


def test_colors_that_do_not_fit_keep_their_hue() -> None:
    green = color.encode_curve(WIDE_GREEN.astype(np.float32))[None, None]
    assert color.out_of_gamut(green, "rec2020", "srgb").all()
    assert not color.out_of_gamut(green, "rec2020", "rec2020").any()
    mapped = color.convert(green, "rec2020", "srgb")[0, 0]
    assert mapped.min() >= 0 and mapped.max() <= 1
    assert mapped[1] > 0.6 and mapped[0] < 0.5
    # Desaturated straight toward its own gray, so the hue holds (clipping the negative
    # channel instead would swing it).
    target = color.convert_linear(WIDE_GREEN.astype(np.float32), "rec2020", "srgb")
    y = float(target @ color.luminance_weights("srgb"))
    got = color.decode_curve(mapped) - y
    assert np.allclose(
        got / np.linalg.norm(got), (target - y) / np.linalg.norm(target - y), atol=0.01
    )


def test_raw_colors_beyond_srgb_survive_editing() -> None:
    img = imaging.decode(make_dng(wide_scene()))
    assert img.space == "rec2020"
    wide = color.out_of_gamut(img.pixels, img.space, "srgb")
    assert wide[:, 70:110].mean() > 0.95 and wide[:, 10:50].mean() == 0

    # A typical edit keeps them: they are only mapped into sRGB on the way out.
    layer = Layer(
        id="L1", name="Look", operations=[ops.Exposure(stops=0.3), ops.Contrast(amount=20)]
    )
    state = EditState(layers=[layer])
    edited = render_state(img.pixels, state, RenderContext(space="rec2020"))
    assert color.out_of_gamut(edited, "rec2020", "srgb")[:, 70:110].mean() > 0.95


def test_sixteen_bit_png_keeps_every_level() -> None:
    ramp = np.linspace(0, 65535, 4096).astype(np.uint16)
    image = np.repeat(np.repeat(ramp[None, :, None], 8, axis=0), 3, axis=2)
    ok, png = cv2.imencode(".png", image)
    assert ok
    img = imaging.decode(png.tobytes())
    assert img.format == "PNG"
    assert len(np.unique(img.pixels[0, :, 0])) > 4000  # 8 bits would leave 256.


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
def test_opens_tiffs(dtype: type) -> None:
    ramp = np.linspace(0, 1, 300, dtype=np.float32)
    rgb = np.stack([ramp, ramp[::-1], np.full_like(ramp, 0.5)], axis=-1)[None].repeat(20, 0)
    if dtype is np.float32:
        stored = color.decode_curve(rgb)  # Float TIFFs hold linear light.
    else:
        stored = np.round(rgb * np.iinfo(dtype).max).astype(dtype)
    buf = io.BytesIO()
    tifffile.imwrite(buf, stored, photometric="rgb", metadata=None)
    img = imaging.decode(buf.getvalue())
    assert img.format == "TIFF"
    assert (img.width, img.height) == (300, 20)
    assert np.abs(img.pixels - rgb).max() < (0.003 if dtype is np.uint8 else 1e-3)


def test_tiff_orientation_and_alpha() -> None:
    rgba = np.zeros((10, 20, 4), np.uint16)
    rgba[..., 0] = 65535
    rgba[..., 3] = 30000
    buf = io.BytesIO()
    tifffile.imwrite(
        buf, rgba, photometric="rgb", extrasamples=["unassalpha"], extratags=[(274, "H", 1, 6, True)]
    )
    img = imaging.decode(buf.getvalue())
    assert (img.width, img.height) == (10, 20)  # Orientation 6: turned a quarter clockwise.
    assert img.alpha is not None and abs(float(img.alpha.mean()) - 30000 / 65535) < 1e-3


def test_previews_of_a_raw_are_srgb(client: TestClient, photos: Path) -> None:
    data = make_dng(wide_scene())
    res = client.post("/api/documents", files={"file": ("green.dng", data)})
    assert res.status_code == 201, res.text
    doc_id = res.json()["id"]
    preview = imaging.decode(client.get(f"/api/documents/{doc_id}/preview").content)
    green = preview.pixels[:, 70:110].mean(axis=(0, 1))
    assert green[1] > 0.6 and green[0] < 0.5  # Mapped into sRGB, still vividly green.
    exported = client.post(f"/api/documents/{doc_id}/export", json={"format": "png"})
    out = imaging.decode(exported.content).pixels[:, 70:110].mean(axis=(0, 1))
    assert np.allclose(out, green, atol=0.03)


def test_uploads_tiffs(client: TestClient) -> None:
    buf = io.BytesIO()
    tifffile.imwrite(buf, np.full((40, 60, 3), 30000, np.uint16), photometric="rgb")
    res = client.post("/api/documents", files={"file": ("scan.tif", buf.getvalue())})
    assert res.status_code == 201, res.text
    assert res.json()["format"] == "TIFF"
