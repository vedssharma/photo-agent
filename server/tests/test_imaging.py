import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from photo_agent import imaging


@pytest.mark.parametrize(
    ("name", "fmt", "size"),
    [
        ("portrait.jpg", "JPEG", (512, 512)),
        ("landscape.png", "PNG", (640, 427)),
        ("low-light.jpg", "JPEG", (600, 400)),
        ("phone.heic", "HEIF", (451, 300)),
    ],
)
def test_decodes_supported_formats(
    photos: Path, name: str, fmt: str, size: tuple[int, int]
) -> None:
    img = imaging.decode((photos / name).read_bytes())
    assert img.format == fmt
    assert (img.width, img.height) == size
    assert img.pixels.dtype == np.float32
    assert img.pixels.shape == (size[1], size[0], 3)
    assert img.pixels.min() >= 0.0 and img.pixels.max() <= 1.0


def test_honors_exif_orientation(photos: Path) -> None:
    upright = imaging.decode((photos / "portrait.jpg").read_bytes()).pixels
    rotated = imaging.decode((photos / "portrait-exif-rotated.jpg").read_bytes()).pixels
    # Same picture once rotated upright, give or take JPEG re-encoding.
    assert float(np.abs(upright - rotated).mean()) < 0.03


def test_keeps_exif_for_export(photos: Path) -> None:
    img = imaging.decode((photos / "phone.heic").read_bytes())
    assert img.exif


def test_rejects_non_images() -> None:
    with pytest.raises(imaging.UnsupportedImageError):
        imaging.decode(b"definitely not a photo")


def test_rejects_unsupported_formats() -> None:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buf, format="GIF")
    with pytest.raises(imaging.UnsupportedImageError, match="GIF"):
        imaging.decode(buf.getvalue())


def test_flattens_alpha_and_remembers_it() -> None:
    buf = io.BytesIO()
    Image.new("RGBA", (4, 4), (255, 0, 0, 128)).save(buf, format="PNG")
    img = imaging.decode(buf.getvalue())
    assert img.has_alpha
    assert img.alpha is not None and img.alpha.shape == (4, 4)
    assert img.pixels.shape == (4, 4, 3)


def test_proxy_downscales_only_large_images() -> None:
    big = np.zeros((2000, 3000, 3), np.float32)
    small = np.zeros((300, 400, 3), np.float32)
    assert imaging.make_proxy(big).shape == (1067, 1600, 3)
    assert imaging.make_proxy(small) is small


def test_jpeg_round_trip(photos: Path) -> None:
    pixels = imaging.decode((photos / "landscape.png").read_bytes()).pixels
    again = imaging.decode(imaging.encode_jpeg(pixels, quality=95)).pixels
    assert float(np.abs(pixels - again).mean()) < 0.02
