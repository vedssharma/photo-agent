"""Exports in TIFF, AVIF, and WebP as well as JPEG and PNG, at 8 or 16 bits."""

import io
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageCms

from photo_agent import imaging
from photo_agent.library import metadata

FORMATS = {"jpeg": "JPEG", "png": "PNG", "tiff": "TIFF", "avif": "AVIF", "webp": "WEBP"}


def _export(client: TestClient, doc_id: str, **options: object) -> bytes:
    res = client.post(f"/api/documents/{doc_id}/export", json=options)
    assert res.status_code == 200, res.text
    return res.content


@pytest.mark.parametrize("fmt", FORMATS)
def test_every_format_keeps_size_profile_and_camera(
    client: TestClient, photos: Path, fmt: str
) -> None:
    data = (photos / "low-light.jpg").read_bytes()
    doc = client.post("/api/documents", files={"file": ("low-light.jpg", data)}).json()
    res = client.post(f"/api/documents/{doc['id']}/export", json={"format": fmt})
    assert res.status_code == 200, res.text
    assert res.headers["content-type"] == f"image/{'tiff' if fmt == 'tiff' else fmt}"
    ext = {"jpeg": "jpg", "tiff": "tif"}.get(fmt, fmt)
    assert res.headers["content-disposition"].endswith(f"low-light-edited.{ext}")
    with Image.open(io.BytesIO(res.content)) as img:
        assert img.format == FORMATS[fmt]
        assert img.size == (600, 400)
        profile = img.info.get("icc_profile")
    assert profile
    assert "sRGB" in ImageCms.getProfileDescription(ImageCms.ImageCmsProfile(io.BytesIO(profile)))
    found = metadata.read(res.content)
    assert (found.camera_make, found.camera_model) == ("Fixture", "Low Light")


def test_lossy_formats_follow_the_quality(client: TestClient, upload: object) -> None:
    doc = upload("landscape.png")  # type: ignore[operator]
    for fmt in ("avif", "webp"):
        small = _export(client, doc["id"], format=fmt, quality=30)
        large = _export(client, doc["id"], format=fmt, quality=95)
        assert len(small) < len(large)


@pytest.mark.parametrize("fmt", ["png", "tiff"])
def test_sixteen_bit_exports_keep_a_raws_gradations(
    client: TestClient, photos: Path, fmt: str
) -> None:
    data = (photos / "raw.dng").read_bytes()
    doc = client.post("/api/documents", files={"file": ("raw.dng", data)}).json()
    eight = imaging.decode(_export(client, doc["id"], format=fmt))
    deep_bytes = _export(client, doc["id"], format=fmt, bit_depth=16, color_space="rec2020")
    deep = imaging.decode(deep_bytes)
    assert deep.space == "rec2020"  # Its profile says so, and the decoder honors it.
    assert len(np.unique(eight.pixels[..., 1])) <= 256
    assert len(np.unique(deep.pixels[..., 1])) > 1000
    found = metadata.read(deep_bytes)
    assert found.camera_model == "RAW Camera"


@pytest.mark.parametrize("fmt", ["png", "tiff", "webp", "avif"])
def test_transparent_cutouts_stay_transparent(client: TestClient, upload: object, fmt: str) -> None:
    doc = upload("portrait.jpg")  # type: ignore[operator]
    state = {
        **doc["state"],
        "cutout": {
            "visible": True,
            "background": None,
            "mask": {"kind": "semantic", "target": "subject"},
        },
    }
    res = client.post(f"/api/documents/{doc['id']}/edits", json={"label": "Cut", "state": state})
    assert res.status_code == 200, res.text
    data = _export(client, doc["id"], format=fmt, bit_depth=16)
    if fmt in ("png", "tiff"):
        alpha = imaging.decode(data).alpha
    else:
        with Image.open(io.BytesIO(data)) as img:
            assert img.mode == "RGBA"
            alpha = np.asarray(img.getchannel("A"), np.float32) / np.float32(255)
    assert alpha is not None
    assert 0.05 < float(alpha.mean()) < 0.95
