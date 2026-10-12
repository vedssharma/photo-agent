"""Photo metadata from EXIF: when it was taken, with what camera and settings, and where.

Read once per photo when it is indexed, from the original file (HEIC included). Location
stays in the library database on this machine; it is never sent to Claude, and exports
strip it by default (see `export.py`).

Places are grouped on a grid of about 10 km (a tenth of a degree), named by their
coordinates: the app has no map data to name them with, and looking places up online would
send the person's whereabouts elsewhere.
"""

from __future__ import annotations

import io
import math
from datetime import datetime
from fractions import Fraction
from typing import Any

from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from photo_agent import imaging

EXIF_IFD = 0x8769
GPS_IFD = 0x8825
MAKE, MODEL = 0x010F, 0x0110
DATETIME = 0x0132
DATETIME_ORIGINAL, DATETIME_DIGITIZED = 0x9003, 0x9004
ISO, EXPOSURE_TIME, F_NUMBER, FOCAL_LENGTH = 0x8827, 0x829A, 0x829D, 0x920A
LENS_MODEL = 0xA434
GPS_LAT_REF, GPS_LAT, GPS_LON_REF, GPS_LON = 1, 2, 3, 4

PLACE_GRID = 0.1
"""Degrees per place cell: photos within roughly 10 km share a place."""


class PhotoMetadata(BaseModel):
    taken_at: datetime | None = None
    """When it was taken, in the camera's local time (EXIF has no time zone)."""
    camera_make: str | None = None
    camera_model: str | None = None
    lens: str | None = None
    iso: int | None = None
    exposure_time: float | None = None
    """Seconds."""
    f_number: float | None = None
    focal_length: float | None = None
    """Millimetres."""
    latitude: float | None = None
    longitude: float | None = None

    @property
    def camera(self) -> str | None:
        return camera_name(self.camera_make, self.camera_model)

    @property
    def place(self) -> str | None:
        if self.latitude is None or self.longitude is None:
            return None
        return place_id(self.latitude, self.longitude)


def camera_name(make: str | None, model: str | None) -> str | None:
    """ "Apple iPhone 15 Pro", not "Apple Apple iPhone 15 Pro" (many models repeat it)."""
    if not model:
        return make or None
    if make and not model.lower().startswith(make.split()[0].lower()):
        return f"{make} {model}"
    return model


def place_id(latitude: float, longitude: float) -> str:
    lat = math.floor(latitude / PLACE_GRID) * PLACE_GRID + PLACE_GRID / 2
    lon = math.floor(longitude / PLACE_GRID) * PLACE_GRID + PLACE_GRID / 2
    return f"{lat:.2f},{lon:.2f}"


def place_label(place: str) -> str:
    lat, lon = (float(v) for v in place.split(","))
    return f"{abs(lat):.1f}° {'N' if lat >= 0 else 'S'}, {abs(lon):.1f}° {'E' if lon >= 0 else 'W'}"


def read(data: bytes) -> PhotoMetadata:
    """Whatever metadata the file has; empty when it has none or cannot be read."""
    try:
        with Image.open(io.BytesIO(data)) as img:
            exif, opened = img.getexif(), True
    except (UnidentifiedImageError, OSError):
        exif, opened = Image.Exif(), False
    if not opened:
        # Camera RAW formats Pillow cannot open still carry TIFF-style EXIF inside.
        raw = imaging.raw_exif(data)
        if not raw:
            return PhotoMetadata()
        exif.load(raw)
    sub = _ifd(exif, EXIF_IFD)
    gps = _ifd(exif, GPS_IFD)
    return PhotoMetadata(
        taken_at=_when(sub.get(DATETIME_ORIGINAL) or sub.get(DATETIME_DIGITIZED))
        or _when(exif.get(DATETIME)),
        camera_make=_text(exif.get(MAKE)),
        camera_model=_text(exif.get(MODEL)),
        lens=_text(sub.get(LENS_MODEL)),
        iso=_int(sub.get(ISO)),
        exposure_time=_number(sub.get(EXPOSURE_TIME)),
        f_number=_number(sub.get(F_NUMBER)),
        focal_length=_number(sub.get(FOCAL_LENGTH)),
        latitude=_coordinate(gps.get(GPS_LAT), gps.get(GPS_LAT_REF), 90),
        longitude=_coordinate(gps.get(GPS_LON), gps.get(GPS_LON_REF), 180),
    )


def _ifd(exif: Image.Exif, tag: int) -> dict[int, Any]:
    try:
        return dict(exif.get_ifd(tag))
    except (KeyError, ValueError, TypeError, OSError):
        return {}


def _text(value: Any) -> str | None:
    if isinstance(value, bytes):
        value = value.decode("utf-8", "replace")
    if not isinstance(value, str):
        return None
    cleaned = value.replace("\x00", "").strip()
    return cleaned[:120] or None


def _number(value: Any) -> float | None:
    if isinstance(value, tuple) and len(value) == 2:  # an unreduced rational
        value = Fraction(int(value[0]), int(value[1])) if value[1] else None
    try:
        number = float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _int(value: Any) -> int | None:
    if isinstance(value, tuple) and value:
        value = value[0]
    number = _number(value)
    return round(number) if number is not None else None


def _when(value: Any) -> datetime | None:
    text = _text(value)
    if not text:
        return None
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y:%m:%d %H:%M"):
        try:
            found = datetime.strptime(text[:19], fmt)
        except ValueError:
            continue
        return found if found.year >= 1900 else None
    return None


def _coordinate(value: Any, ref: Any, limit: float) -> float | None:
    if not isinstance(value, tuple) or len(value) != 3:
        return None
    parts = [_number(v) if v != 0 else 0.0 for v in value]
    if any(p is None for p in parts):
        return None
    degrees, minutes, seconds = (float(p) for p in parts)  # type: ignore[arg-type]
    decimal = degrees + minutes / 60 + seconds / 3600
    if _text(ref) in ("S", "W"):
        decimal = -decimal
    if not -limit <= decimal <= limit or (decimal == 0 and degrees == 0 and minutes == 0):
        return None
    return round(decimal, 6)
