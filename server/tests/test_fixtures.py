"""The shared test photos decode and match what fixtures/photos/manifest.json says."""

import json
from typing import Any

import pillow_heif
import pytest
from PIL import Image, ImageOps

from photo_agent.settings import REPO_ROOT

pillow_heif.register_heif_opener()

PHOTOS = REPO_ROOT / "fixtures" / "photos"
MANIFEST: list[dict[str, Any]] = json.loads((PHOTOS / "manifest.json").read_text())

GPS_IFD = 0x8825
EXIF_ORIENTATION = 0x0112


def test_manifest_covers_required_kinds() -> None:
    formats = {entry["format"] for entry in MANIFEST}
    tags = {tag for entry in MANIFEST for tag in entry["tags"]}
    assert {"JPEG", "PNG", "HEIF"} <= formats
    assert {"portrait", "landscape", "low-light"} <= tags


@pytest.mark.parametrize("entry", MANIFEST, ids=lambda e: e["file"])
def test_fixture_decodes_as_described(entry: dict[str, Any]) -> None:
    with Image.open(PHOTOS / entry["file"]) as img:
        assert img.format == entry["format"]
        assert img.getexif().get(EXIF_ORIENTATION, 1) == entry.get("orientation", 1)
        assert (GPS_IFD in img.getexif()) == entry.get("has_gps", False)

        upright = ImageOps.exif_transpose(img)
        assert upright.size == (entry["width"], entry["height"])
        assert upright.convert("RGB").getbbox() is not None


def test_every_photo_is_in_the_manifest() -> None:
    on_disk = {p.name for p in PHOTOS.iterdir() if p.suffix != ".json"}
    assert on_disk == {entry["file"] for entry in MANIFEST}
