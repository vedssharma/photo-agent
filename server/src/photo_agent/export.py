"""Export: render the edit graph at full resolution and encode it for download."""

from __future__ import annotations

from typing import Literal

import cv2
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

from photo_agent import imaging
from photo_agent.graph import Document
from photo_agent.render import render_cutout, render_state
from photo_agent.store import LoadedImage
from photo_agent.vision.selection import upscale_plainly

EXIF_ORIENTATION = 0x0112
EXIF_IFD = 0x8769
GPS_IFD = 0x8825
EXIF_PIXEL_X = 0xA002
EXIF_PIXEL_Y = 0xA003


class ExportOptions(BaseModel):
    format: Literal["jpeg", "png"] = "jpeg"
    quality: int = Field(92, ge=1, le=100, description="JPEG quality; ignored for PNG.")
    keep_location: bool = Field(
        False, description="Keep GPS location from the original. Off by default for privacy."
    )
    upscale: Literal[1, 2, 4] = Field(
        1, description="Enlarge the export with an AI upscaler (up to 8000 pixels across)."
    )


MAX_UPSCALED_EDGE = 8000
"""The longest side an upscaled export can have, to keep files and memory sensible."""


TRANSPARENT_JPEG_BACKGROUND = "#ffffff"
"""JPEG has no transparency, so a transparent cutout exported as JPEG goes on white."""


def export_bytes(doc: Document, loaded: LoadedImage, options: ExportOptions) -> bytes:
    state, ctx = doc.state, loaded.full_context
    cutout = state.cutout
    transparent = cutout is not None and cutout.visible and cutout.background is None
    if transparent and options.format == "png":
        pixels, alpha = render_cutout(loaded.source.pixels, state, ctx)
    else:
        if transparent and cutout is not None:
            white = cutout.model_copy(update={"background": TRANSPARENT_JPEG_BACKGROUND})
            state = state.model_copy(update={"cutout": white})
        pixels, alpha = render_state(loaded.source.pixels, state, ctx), None
    if options.upscale > 1:
        pixels, alpha = enlarge(loaded, pixels, alpha, options.upscale)
    exif = export_exif(loaded.source.exif, pixels.shape[1], pixels.shape[0], options.keep_location)
    if options.format == "png":
        return imaging.encode_png(pixels, exif=exif, alpha=alpha)
    return imaging.encode_jpeg(pixels, quality=options.quality, exif=exif)


def enlarge(
    loaded: LoadedImage, pixels: imaging.Array, alpha: imaging.Array | None, upscale: int
) -> tuple[imaging.Array, imaging.Array | None]:
    """The export enlarged by `upscale`, or less if that would pass MAX_UPSCALED_EDGE."""
    h, w = pixels.shape[:2]
    scale = min(float(upscale), MAX_UPSCALED_EDGE / max(h, w))
    if scale <= 1.01:
        return pixels, alpha
    vision = loaded.vision
    big = vision.upscale(pixels, scale) if vision else upscale_plainly(pixels, scale)
    if alpha is not None:
        size = big.shape[1::-1]
        alpha = np.asarray(cv2.resize(alpha, size, interpolation=cv2.INTER_LINEAR), np.float32)
    return big, alpha


def export_exif(raw: bytes, width: int, height: int, keep_location: bool) -> bytes | None:
    """Carry camera metadata over to the export, fixed up for the edited pixels.

    The pixels are already upright, so orientation resets to normal; the recorded size
    becomes the exported size; and GPS location is dropped unless asked for.
    """
    if not raw:
        return None
    exif = Image.Exif()
    try:
        exif.load(raw)
    except Exception:  # Unreadable EXIF is not worth failing an export over.
        return None
    exif[EXIF_ORIENTATION] = 1
    if not keep_location and GPS_IFD in exif:
        del exif[GPS_IFD]
    sub = exif.get_ifd(EXIF_IFD)
    if sub:
        sub[EXIF_PIXEL_X] = width
        sub[EXIF_PIXEL_Y] = height
    return exif.tobytes()


def export_filename(doc: Document, options: ExportOptions) -> str:
    stem = doc.filename.rsplit(".", 1)[0] or "photo"
    return f"{stem}-edited.{'jpg' if options.format == 'jpeg' else 'png'}"
