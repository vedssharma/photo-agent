"""Builders shared by the library tests."""

from pathlib import Path
from typing import Any

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

PHOTOS = ["landscape.png", "portrait.jpg", "low-light.jpg"]


def files(photos: Path, *names: str) -> list[tuple[str, tuple[str, bytes]]]:
    return [("files", (name, (photos / name).read_bytes())) for name in names]


def mean(image: Image.Image) -> float:
    return float(np.asarray(image.convert("L"), np.float32).mean())


def import_photos(client: TestClient, photos: Path, *names: str) -> dict[str, Any]:
    res = client.post("/api/library/photos", files=files(photos, *names))
    assert res.status_code == 201, res.text
    body: dict[str, Any] = res.json()
    return body


def jpeg_with_exif(
    *,
    taken: str | None = None,
    make: str | None = None,
    model: str | None = None,
    gps: tuple[float, float] | None = None,
    color: tuple[int, int, int] = (120, 140, 160),
    size: tuple[int, int] = (64, 48),
    **settings: Any,
) -> bytes:
    """A small JPEG with the given EXIF: DateTimeOriginal "YYYY:MM:DD HH:MM:SS", camera,
    GPS (latitude, longitude), and settings (iso, exposure_time, f_number, focal_length,
    lens)."""
    import io
    from fractions import Fraction

    from PIL.TiffImagePlugin import IFDRational

    def rational(v: float) -> IFDRational:
        f = Fraction(v).limit_denominator(10000)
        return IFDRational(f.numerator, f.denominator)

    img = Image.new("RGB", size, color)
    exif = Image.Exif()
    if make:
        exif[0x010F] = make
    if model:
        exif[0x0110] = model
    sub = exif.get_ifd(0x8769)
    if taken:
        sub[0x9003] = taken
    tags = {"iso": 0x8827, "exposure_time": 0x829A, "f_number": 0x829D, "focal_length": 0x920A}
    for name, tag in tags.items():
        if name in settings:
            value = settings[name]
            sub[tag] = value if name == "iso" else rational(value)
    if "lens" in settings:
        sub[0xA434] = settings["lens"]
    if gps:
        lat, lon = gps

        def dms(v: float) -> tuple[IFDRational, IFDRational, IFDRational]:
            v = abs(v)
            d = int(v)
            m = int((v - d) * 60)
            s = (v - d - m / 60) * 3600
            return rational(d), rational(m), rational(s)

        g = exif.get_ifd(0x8825)
        g[1], g[2] = ("N" if lat >= 0 else "S"), dms(lat)
        g[3], g[4] = ("E" if lon >= 0 else "W"), dms(lon)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)
    return buf.getvalue()


def import_bytes(client: TestClient, *named: tuple[str, bytes]) -> list[dict[str, Any]]:
    res = client.post("/api/library/photos", files=[("files", item) for item in named])
    assert res.status_code == 201, res.text
    imported: list[dict[str, Any]] = res.json()["imported"]
    return imported
