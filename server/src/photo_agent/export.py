"""Export: render the edit graph at full resolution and encode it for download."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

from photo_agent import color, credentials, icc, imaging
from photo_agent.graph import Document
from photo_agent.render import render_cutout, render_state
from photo_agent.store import LoadedImage
from photo_agent.vision.selection import upscale_plainly

EXIF_ORIENTATION = 0x0112
EXIF_IFD = 0x8769
GPS_IFD = 0x8825
EXIF_PIXEL_X = 0xA002
EXIF_PIXEL_Y = 0xA003


ExportFormat = Literal["jpeg", "png", "tiff", "avif", "webp"]

EXTENSIONS: dict[str, str] = {
    "jpeg": "jpg",
    "png": "png",
    "tiff": "tif",
    "avif": "avif",
    "webp": "webp",
}
MEDIA_TYPES: dict[str, str] = {
    "jpeg": "image/jpeg",
    "png": "image/png",
    "tiff": "image/tiff",
    "avif": "image/avif",
    "webp": "image/webp",
}
LOSSY = ("jpeg", "avif", "webp")
"""Formats that take a quality setting."""
DEEP = ("png", "tiff")
"""Formats that can hold 16 bits per channel."""
TRANSPARENT = ("png", "tiff", "avif", "webp")
"""Formats that can hold a transparent background."""


class ExportOptions(BaseModel):
    format: ExportFormat = Field(
        "jpeg",
        description="JPEG for sharing; PNG lossless; TIFF lossless for printing or further "
        "editing; AVIF and WebP for the web, smaller than JPEG at the same quality.",
    )
    quality: int = Field(
        92, ge=1, le=100, description="Quality for JPEG, AVIF, and WebP; ignored otherwise."
    )
    bit_depth: Literal[8, 16] = Field(
        8,
        description="Bits per color channel for PNG and TIFF; 16 keeps the smooth "
        "gradations of RAW and 16-bit photos for further editing. Other formats are 8.",
    )
    keep_location: bool = Field(
        False, description="Keep GPS location from the original. Off by default for privacy."
    )
    upscale: Literal[1, 2, 4] = Field(
        1, description="Enlarge the export with an AI upscaler (up to 8000 pixels across)."
    )
    color_space: color.Space = Field(
        "srgb",
        description="The color space the file is saved in, with its ICC profile embedded. "
        "sRGB suits sharing and the web; Display P3 keeps the extra color of modern phone "
        "and laptop screens; Rec.2020 keeps every color a RAW file holds, for further "
        "editing (best with 16 bits).",
    )


MAX_UPSCALED_EDGE = 8000
"""The longest side an upscaled export can have, to keep files and memory sensible."""


TRANSPARENT_JPEG_BACKGROUND = "#ffffff"
"""JPEG has no transparency, so a transparent cutout exported as JPEG goes on white."""


def export_bytes(
    doc: Document, loaded: LoadedImage, options: ExportOptions, data_dir: Path
) -> bytes:
    """The encoded export, with Content Credentials when generative edits show in it."""
    data = _encode(doc, loaded, options)
    if credentials.visible_generative_ops(doc.state):
        data = credentials.sign(
            data, options.format, doc.state, export_filename(doc, options), data_dir
        )
    return data


def _encode(doc: Document, loaded: LoadedImage, options: ExportOptions) -> bytes:
    state, ctx = doc.state, loaded.full_context
    cutout = state.cutout
    transparent = cutout is not None and cutout.visible and cutout.background is None
    if transparent and options.format in TRANSPARENT:
        pixels, alpha = render_cutout(loaded.source.pixels, state, ctx)
    else:
        if transparent and cutout is not None:
            white = cutout.model_copy(update={"background": TRANSPARENT_JPEG_BACKGROUND})
            state = state.model_copy(update={"cutout": white})
        pixels, alpha = render_state(loaded.source.pixels, state, ctx), None
    if options.upscale > 1:
        pixels, alpha = enlarge(loaded, pixels, alpha, options.upscale)
    # The output transform: into the chosen space, with colors that do not fit mapped
    # onto its edge, and that space's profile embedded.
    pixels = color.convert(pixels, loaded.space, options.color_space)
    profile = icc.profile(options.color_space)
    exif = export_exif(loaded.source.exif, pixels.shape[1], pixels.shape[0], options.keep_location)
    bits = options.bit_depth if options.format in DEEP else 8
    if options.format == "png":
        return imaging.encode_png(pixels, exif=exif, alpha=alpha, profile=profile, bits=bits)
    if options.format == "tiff":
        return imaging.encode_tiff(pixels, exif=exif, alpha=alpha, profile=profile, bits=bits)
    if options.format in ("avif", "webp"):
        fmt = options.format.upper()
        return imaging.encode_pillow(pixels, fmt, options.quality, exif, alpha, profile)
    return imaging.encode_jpeg(pixels, quality=options.quality, exif=exif, profile=profile)


def enlarge(
    loaded: LoadedImage, pixels: imaging.Array, alpha: imaging.Array | None, upscale: int
) -> tuple[imaging.Array, imaging.Array | None]:
    """The export enlarged by `upscale`, or less if that would pass MAX_UPSCALED_EDGE."""
    h, w = pixels.shape[:2]
    scale = min(float(upscale), MAX_UPSCALED_EDGE / max(h, w))
    if scale <= 1.01:
        return pixels, alpha
    vision = loaded.vision
    if vision is None:
        big = upscale_plainly(pixels, scale)
    elif loaded.space == "srgb":
        big = vision.upscale(pixels, scale)
    else:
        # The upscaler only knows sRGB photos, so the photo is enlarged plainly in its own
        # space and given the fine detail the upscaler adds to its sRGB version.
        srgb = loaded.to_srgb(pixels)
        detail = vision.upscale(srgb, scale) - upscale_plainly(srgb, scale)
        big = np.asarray(np.clip(upscale_plainly(pixels, scale) + detail, 0, 1), np.float32)
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
    return f"{stem}-edited.{EXTENSIONS[options.format]}"
