"""HDR merges and panoramas: several photos combined into a new one."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from raw_helpers import make_dng, srgb_to_linear_rec2020

from photo_agent import color, imaging, merge
from photo_agent.library import metadata


@pytest.fixture
def scene(photos: Path) -> imaging.DecodedImage:
    return imaging.decode((photos / "landscape.png").read_bytes())


def exposed(scene: imaging.DecodedImage, stops: float, shift: int = 0) -> imaging.DecodedImage:
    """The scene shot `stops` brighter or darker, optionally a few pixels to the side."""
    linear = color.decode_curve(scene.pixels) * 2.0**stops
    pixels = color.encode_curve(np.clip(linear, 0, 1))
    if shift:
        pixels = np.ascontiguousarray(np.roll(pixels, shift, axis=1))
    return imaging.DecodedImage(pixels=pixels, format="PNG")


def tiles(scene: imaging.DecodedImage) -> list[imaging.DecodedImage]:
    """Three overlapping views across the scene, as if turning the camera."""
    return [
        imaging.DecodedImage(pixels=np.ascontiguousarray(scene.pixels[:, a:b]), format="PNG")
        for a, b in ((0, 360), (140, 500), (280, 640))
    ]


def test_hdr_keeps_detail_in_shadows_and_highlights(scene: imaging.DecodedImage) -> None:
    bracket = [exposed(scene, -2), exposed(scene, 0), exposed(scene, 2, shift=4)]
    fused, space = merge.merge_hdr(bracket)
    assert space == "srgb" and fused.shape == scene.pixels.shape
    luma = fused.mean(axis=2)
    bright = bracket[2].pixels.mean(axis=2)
    dark = bracket[0].pixels.mean(axis=2)
    # Fewer blown highlights than the bright frame, fewer crushed shadows than the dark one.
    assert (luma > 0.98).mean() < (bright > 0.98).mean() / 2
    assert (luma < 0.02).mean() < (dark < 0.02).mean() / 2
    # The hand-held frame was lined up, so edges stay crisp (no doubled image).
    aligned, _ = merge.merge_hdr([exposed(scene, -2), exposed(scene, 0), exposed(scene, 2)])
    assert float(np.abs(fused - aligned)[:, 10:-10].mean()) < 0.01


def test_hdr_needs_one_scene(scene: imaging.DecodedImage, photos: Path) -> None:
    other = imaging.decode((photos / "portrait.jpg").read_bytes())
    with pytest.raises(merge.MergeError, match="different sizes"):
        merge.merge_hdr([scene, other])
    with pytest.raises(merge.MergeError, match="at least two"):
        merge.merge_hdr([scene])


def test_stitches_overlapping_photos(scene: imaging.DecodedImage) -> None:
    pano, _ = merge.stitch_panorama(tiles(scene))
    h, w = pano.shape[:2]
    assert w > 560 and h > 380  # Much wider than any one tile (360), barely cropped.
    # It shows the scene: compare with the matching middle part of the original.
    middle = pano[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4].mean(axis=(0, 1))
    sh, sw = scene.pixels.shape[:2]
    original = scene.pixels[sh // 4 : 3 * sh // 4, sw // 4 : 3 * sw // 4].mean(axis=(0, 1))
    assert np.allclose(middle, original, atol=0.05)
    assert pano.min() >= 0 and (pano.max(axis=2) > 0).all()  # No empty corners left.


def test_unrelated_photos_do_not_stitch(scene: imaging.DecodedImage, photos: Path) -> None:
    other = imaging.decode((photos / "portrait.jpg").read_bytes())
    with pytest.raises(merge.MergeError, match="overlap"):
        merge.stitch_panorama([scene, other])


def test_raw_brackets_merge_into_a_wide_16_bit_tiff(scene: imaging.DecodedImage) -> None:
    light = srgb_to_linear_rec2020(scene.pixels)
    sources = [make_dng(light, exposure=2.0**ev, make="Canon", model="R5") for ev in (-1, 1)]
    data, space = merge.merge("hdr", sources)
    assert space == "rec2020"
    out = imaging.decode(data)
    assert (out.format, out.space) == ("TIFF", "rec2020")
    assert len(np.unique(out.pixels[..., 1])) > 4096  # More levels than 8 bits could hold.
    found = metadata.read(data)
    assert (found.camera_make, found.camera_model) == ("Canon", "R5")
    assert found.taken_at is not None


def _import(client: TestClient, files: list[tuple[str, bytes]]) -> list[dict[str, Any]]:
    res = client.post("/api/library/photos", files=[("files", f) for f in files])
    assert res.status_code == 201, res.text
    imported: list[dict[str, Any]] = res.json()["imported"]
    return imported


def test_merge_in_the_library(client: TestClient, scene: imaging.DecodedImage) -> None:
    parts = [(f"pano-{i}.png", imaging.encode_png(t.pixels)) for i, t in enumerate(tiles(scene))]
    ids = [p["id"] for p in _import(client, parts)]
    res = client.post("/api/library/merge", json={"ids": ids, "kind": "panorama"})
    assert res.status_code == 201, res.text
    made = res.json()
    assert made["filename"] == "pano-0-panorama.tif"
    assert made["format"] == "TIFF" and made["width"] > 560
    listed = {p["id"] for p in client.get("/api/library/photos").json()}
    assert made["id"] in listed and set(ids) <= listed

    # The same merge again would be a copy of the first.
    again = client.post("/api/library/merge", json={"ids": ids, "kind": "panorama"})
    assert again.status_code == 409

    hdr = client.post("/api/library/merge", json={"ids": [ids[0], made["id"]], "kind": "hdr"})
    assert hdr.status_code == 422  # Different sizes: not a bracket.
    assert "different sizes" in hdr.json()["detail"]
    missing = client.post("/api/library/merge", json={"ids": [ids[0], "f" * 12], "kind": "hdr"})
    assert missing.status_code == 404
