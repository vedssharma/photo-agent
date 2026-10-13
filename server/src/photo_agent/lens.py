"""Lens profile corrections from the Lensfun database.

Lensfun has measured profiles for over a thousand lenses: how much each bends straight
lines (distortion), darkens toward the corners (vignetting), and splits colors at edges
(transverse chromatic aberration), at each focal length and aperture. The camera and lens
come from the photo's EXIF, so a `lens_profile` operation needs no settings to do the
right thing, and the same operation in a recipe corrects each photo for its own lens.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import cv2
import lensfunpy
import numpy as np
import numpy.typing as npt
from PIL import Image
from pydantic import BaseModel

from photo_agent import color
from photo_agent import operations as ops

Array = npt.NDArray[np.float32]

EXIF_IFD = 0x8769
MAKE, MODEL = 0x010F, 0x0110
LENS_MAKE, LENS_MODEL = 0xA433, 0xA434
FOCAL_LENGTH, F_NUMBER, FOCAL_35MM = 0x920A, 0x829D, 0xA405
SUBJECT_DISTANCE = 0x9206

DEFAULT_DISTANCE = 1000.0
"""Focus distance (meters) when the photo does not say: effectively infinity."""


@dataclass(frozen=True)
class Shot:
    """What the photo's EXIF says about the camera and lens it was taken with."""

    make: str = ""
    model: str = ""
    lens: str = ""
    focal_length: float = 0.0
    aperture: float = 0.0
    focal_35mm: float = 0.0
    distance: float = DEFAULT_DISTANCE


def shot_from_exif(raw: bytes) -> Shot | None:
    """The camera and lens from EXIF bytes; None without a focal length to go on."""
    if not raw:
        return None
    exif = Image.Exif()
    try:
        exif.load(raw)
        sub = dict(exif.get_ifd(EXIF_IFD))
    except Exception:  # Unreadable EXIF just means no profile.
        return None

    def text(v: Any) -> str:
        return v.strip("\0 ").strip() if isinstance(v, str) else ""

    def number(v: Any) -> float:
        try:
            return float(v)
        except (TypeError, ValueError, ZeroDivisionError):
            return 0.0

    shot = Shot(
        make=text(exif.get(MAKE)),
        model=text(exif.get(MODEL)),
        lens=text(sub.get(LENS_MODEL)),
        focal_length=number(sub.get(FOCAL_LENGTH)),
        aperture=number(sub.get(F_NUMBER)),
        focal_35mm=number(sub.get(FOCAL_35MM)),
        distance=number(sub.get(SUBJECT_DISTANCE)) or DEFAULT_DISTANCE,
    )
    return shot if shot.focal_length > 0 else None


@lru_cache(maxsize=1)
def database() -> lensfunpy.Database:
    """The bundled Lensfun database (loaded once; it takes a moment)."""
    return lensfunpy.Database()


@dataclass(frozen=True)
class Match:
    lens: Any
    """The Lensfun lens."""
    camera: str
    crop_factor: float


_lock = threading.Lock()


def find(shot: Shot, lens_name: str = "") -> Match | None:
    """The Lensfun profile for the shot's lens (or `lens_name`, to override it)."""
    wanted = lens_name.strip() or shot.lens
    if not wanted:
        return None
    return _find(shot.make, shot.model, wanted, shot.focal_length, shot.focal_35mm)


@lru_cache(maxsize=64)
def _find(make: str, model: str, wanted: str, focal: float, focal_35mm: float) -> Match | None:
    db = database()
    with _lock:
        cameras = db.find_cameras(make, model, loose_search=True) if make else []
        if not cameras and make:
            # A camera too new for the database: any body of the same make shares its
            # mounts, which is what the lens search needs.
            cameras = db.find_cameras(make, None, loose_search=True)
        found = None
        for camera in cameras[:8]:
            lenses = db.find_lenses(camera, None, wanted, loose_search=True)
            if lenses:
                found = (camera, lenses[0])
                break
    if found is None:
        return None
    camera, lens = found
    exact = camera.model == model
    crop = float(camera.crop_factor)
    if not exact and focal_35mm > 0 and focal > 0:
        crop = focal_35mm / focal  # The body was a stand-in; trust the photo's own ratio.
    return Match(lens=lens, camera=model or camera.model, crop_factor=crop)


class LensView(BaseModel):
    """What is known about the lens a photo was taken with, and its profile."""

    camera: str | None = None
    lens: str | None = None
    """The lens as the photo's EXIF names it."""
    focal_length: float | None = None
    aperture: float | None = None
    profile: str | None = None
    """The Lensfun profile that corrects it, or None when there is none."""


def describe(shot: Shot | None, lens_name: str = "") -> LensView:
    if shot is None:
        return LensView()
    found = find(shot, lens_name)
    return LensView(
        camera=" ".join(p for p in (shot.make, shot.model) if p) or None,
        lens=shot.lens or None,
        focal_length=shot.focal_length or None,
        aperture=shot.aperture or None,
        profile=f"{found.lens.maker} {found.lens.model}".strip() if found else None,
    )


# Correcting


_maps: OrderedDict[tuple[Any, ...], tuple[Array | None, Array | None]] = OrderedDict()
_maps_lock = threading.Lock()


def _coordinates(found: Match, shot: Shot, w: int, h: int) -> tuple[Array | None, Array | None]:
    """Where each output pixel samples the photo: for distortion (h, w, 2) and for
    chromatic aberration per channel (h, w, 3, 2); None where the profile has no data."""
    key = (id(found.lens), found.crop_factor, w, h, shot.focal_length, shot.aperture)
    with _maps_lock:
        if key in _maps:
            _maps.move_to_end(key)
            return _maps[key]

    def modifier(flags: int) -> lensfunpy.Modifier:
        m = lensfunpy.Modifier(found.lens, found.crop_factor, w, h)
        m.initialize(
            shot.focal_length,
            shot.aperture or 8.0,
            shot.distance,
            scale=0.0,  # Automatic: just enough zoom that no edge is left empty.
            pixel_format=np.float32,
            flags=flags,
        )
        return m

    geometry = lensfunpy.ModifyFlags.DISTORTION | lensfunpy.ModifyFlags.SCALE
    distortion = modifier(geometry).apply_geometry_distortion()
    tca = modifier(lensfunpy.ModifyFlags.TCA).apply_subpixel_distortion()
    result = (
        None if distortion is None else np.asarray(distortion, np.float32),
        None if tca is None else np.asarray(tca, np.float32),
    )
    with _maps_lock:
        _maps[key] = result
        while len(_maps) > 4:
            _maps.popitem(last=False)
    return result


def correct(
    x: Array, op: ops.LensProfile, shot: Shot | None, space: color.WorkingSpace = "srgb"
) -> Array:
    """`x` corrected for its lens by `op`'s amounts; unchanged when there is no profile."""
    if shot is None:
        return x
    found = find(shot, op.lens)
    if found is None:
        return x
    h, w = x.shape[:2]
    out = x
    if op.vignetting > 0:
        m = lensfunpy.Modifier(found.lens, found.crop_factor, w, h)
        m.initialize(
            shot.focal_length,
            shot.aperture or 8.0,
            shot.distance,
            pixel_format=np.float32,
            flags=lensfunpy.ModifyFlags.VIGNETTING,
        )
        # Vignetting is a loss of light, so it is undone on linear values.
        linear = color.decode_curve(out)
        brightened = np.ascontiguousarray(linear, dtype=np.float32)
        if m.apply_color_modification(brightened):
            mixed = linear + (brightened - linear) * (op.vignetting / 100)
            out = color.encode_curve(np.asarray(mixed, np.float32))
    if op.distortion <= 0 and op.chromatic_aberration <= 0:
        return out
    distortion, tca = _coordinates(found, shot, w, h)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    identity = np.stack([xs, ys], axis=-1)
    where = np.repeat(identity[:, :, None, :], 3, axis=2)
    if distortion is not None and op.distortion > 0:
        where = where + ((distortion - identity) * (op.distortion / 100))[:, :, None, :]
    if tca is not None and op.chromatic_aberration > 0:
        where = where + (tca - identity[:, :, None, :]) * (op.chromatic_aberration / 100)
    channels = [
        cv2.remap(
            np.ascontiguousarray(out[..., c]),
            np.ascontiguousarray(where[:, :, c, 0]),
            np.ascontiguousarray(where[:, :, c, 1]),
            cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        for c in range(3)
    ]
    return np.asarray(np.stack(channels, axis=-1), dtype=np.float32)
