"""Camera RAW files open like any other photo: developed by LibRaw, with their EXIF."""

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from raw_helpers import make_dng, srgb_to_linear_rec2020

from photo_agent import imaging
from photo_agent.library import metadata
from photo_agent.settings import Settings


@pytest.fixture
def scene(photos: Path) -> np.ndarray:
    return np.asarray(Image.open(photos / "landscape.png").convert("RGB"), np.float32) / 255


def test_develops_a_raw_file_into_the_scene_it_shot(scene: np.ndarray) -> None:
    img = imaging.decode(make_dng(srgb_to_linear_rec2020(scene)))
    assert img.format == imaging.RAW
    assert img.pixels.dtype == np.float32
    h, w = img.pixels.shape[:2]
    assert (w, h) == (640, 426)  # Bayer mosaics have even sizes.
    # Demosaicing blurs the sharpest edges a little; colors and tones match.
    assert float(np.abs(img.pixels - scene[:h, :w]).mean()) < 0.02
    assert np.allclose(img.pixels.mean(axis=(0, 1)), scene[:h, :w].mean(axis=(0, 1)), atol=0.01)


def test_the_fixture_raw_opens(photos: Path) -> None:
    img = imaging.decode((photos / "raw.dng").read_bytes())
    assert (img.width, img.height) == (600, 400)
    # A warm indoor scene (coffee), so white balance and the color matrix were applied.
    r, g, b = img.pixels.mean(axis=(0, 1))
    assert r > g > b


def test_recovers_clipped_highlights_without_color_casts() -> None:
    # A neutral ramp that runs well past the sensor's white level. Green clips first
    # (the sensor is most sensitive to it); without reconstruction that turns highlights
    # magenta once the white balance lifts red and blue.
    ramp = np.linspace(0.0, 3.0, 400)
    scene = np.repeat(np.repeat(ramp[None, :, None], 40, axis=0), 3, axis=2)
    img = imaging.decode(make_dng(scene)).pixels
    row = img[20, 2:-2]
    chroma = row.max(axis=1) - row.min(axis=1)
    assert float(chroma.max()) < 0.05
    # Tones keep rising past where green clipped (scene 1.0), instead of a flat white.
    at = {v: int(np.argmin(np.abs(ramp[2:-2] - v))) for v in (0.5, 1.2, 2.0)}
    assert row[at[0.5], 1] < row[at[1.2], 1] < row[at[2.0], 1] <= 1.0


def test_reads_camera_exif_from_raw(scene: np.ndarray) -> None:
    data = make_dng(srgb_to_linear_rec2020(scene), make="Canon", model="Canon EOS 5D Mark III")
    found = metadata.read(data)
    assert found.camera_make == "Canon"
    assert found.camera_model == "Canon EOS 5D Mark III"
    assert found.taken_at is not None and found.taken_at.year == 2026
    exif = Image.Exif()
    exif.load(imaging.decode(data).exif)
    assert exif[0x0112] == 1  # LibRaw already turned it upright.


def test_raw_files_are_not_mistaken_for_other_things() -> None:
    with pytest.raises(imaging.UnsupportedImageError):
        imaging.decode(b"II*\x00" + bytes(200))


def test_upload_and_export_a_raw(client: TestClient, photos: Path, settings: Settings) -> None:
    data = (photos / "raw.dng").read_bytes()
    res = client.post("/api/documents", files={"file": ("IMG_0001.dng", data)})
    assert res.status_code == 201, res.text
    doc = res.json()
    assert doc["format"] == "RAW"
    assert (doc["width"], doc["height"]) == (600, 400)
    folder = settings.data_dir / "documents" / doc["id"]
    assert (folder / "original.dng").read_bytes() == data

    res = client.post(f"/api/documents/{doc['id']}/export", json={"format": "jpeg"})
    assert res.status_code == 200
    out = imaging.decode(res.content)
    assert (out.width, out.height) == (600, 400)
