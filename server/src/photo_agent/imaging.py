"""Image ingest and encoding: decode uploads into the working color space and back.

The working representation is an RGB float32 NumPy array of shape (height, width, 3),
sRGB-encoded with values in [0, 1]. Every decoder output is rotated upright per its EXIF
orientation, so the rest of the pipeline never thinks about orientation again.

Camera RAW files are developed by LibRaw (through `rawpy`): demosaiced, white balanced as
shot, converted from the camera's own color response with its color matrix, and with
clipped highlights reconstructed rather than cut off.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import cv2
import numpy as np
import numpy.typing as npt
import pillow_heif
import rawpy
from PIL import Image, ImageCms, ImageOps, UnidentifiedImageError

pillow_heif.register_heif_opener()

Array = npt.NDArray[np.float32]

SUPPORTED_FORMATS = {"JPEG", "PNG", "HEIF"}
"""Pillow format names we accept. `pillow-heif` reports HEIC files as HEIF."""

RAW = "RAW"
"""The format name for camera RAW files, which LibRaw rather than Pillow reads."""

RAW_EXTENSIONS = [
    ".dng",
    ".cr2",
    ".cr3",
    ".crw",
    ".nef",
    ".nrw",
    ".arw",
    ".srf",
    ".sr2",
    ".raf",
    ".orf",
    ".rw2",
    ".pef",
    ".srw",
    ".x3f",
    ".3fr",
    ".iiq",
    ".rwl",
    ".erf",
    ".mrw",
    ".kdc",
    ".dcr",
    ".mos",
]
"""File extensions of the RAW formats we offer to open (LibRaw reads many more)."""

FORMAT_ALIASES = {"MPO": "JPEG"}
"""Pillow formats we treat as another. iPhone JPEGs carry a depth map or gain map as a
second image, so Pillow calls them MPO; the first frame is the ordinary JPEG photo."""

PREVIEW_LONG_EDGE = 1600
"""Long edge of the preview proxy, in pixels. Interactive renders happen at this size."""

EXIF_ORIENTATION = 0x0112
EXIF_IFD_TAG = 0x8769
GPS_IFD_TAG = 0x8825
SRGB_PROFILE = ImageCms.createProfile("sRGB")
SRGB_ICC = ImageCms.ImageCmsProfile(SRGB_PROFILE).tobytes()


class UnsupportedImageError(ValueError):
    """The upload is not an image we can open."""


@dataclass
class DecodedImage:
    pixels: Array
    """Upright RGB float32 pixels in [0, 1], sRGB-encoded."""
    format: str
    """Source format as Pillow names it: JPEG, PNG, or HEIF."""
    exif: bytes = b""
    """Raw EXIF from the source (orientation not yet reset), or empty."""
    has_alpha: bool = False
    alpha: Array | None = field(default=None, repr=False)
    """Upright alpha channel in [0, 1] when the source had one."""

    @property
    def width(self) -> int:
        return int(self.pixels.shape[1])

    @property
    def height(self) -> int:
        return int(self.pixels.shape[0])


def decode(data: bytes) -> DecodedImage:
    """Decode JPEG, PNG, HEIC, or camera RAW bytes into upright sRGB float32 pixels."""
    try:
        img = Image.open(io.BytesIO(data))
        fmt = FORMAT_ALIASES.get(img.format or "", img.format)
        if fmt == "TIFF":
            # Many RAW formats are TIFF underneath, so Pillow claims them; ask LibRaw.
            return decode_raw(data)
        img.load()
    except UnsupportedImageError:
        raise
    except (UnidentifiedImageError, OSError) as exc:
        try:
            return decode_raw(data)
        except UnsupportedImageError:
            raise UnsupportedImageError("Could not read this file as an image.") from exc
    if fmt not in SUPPORTED_FORMATS:
        raise UnsupportedImageError(
            f"{img.format or 'This'} files are not supported yet; "
            "use JPEG, PNG, HEIC, or a camera RAW file."
        )
    exif = img.info.get("exif", b"")
    icc = img.info.get("icc_profile")

    upright = ImageOps.exif_transpose(img)
    alpha: Array | None = None
    if upright.mode in ("RGBA", "LA") or (upright.mode == "P" and "transparency" in upright.info):
        rgba = upright.convert("RGBA")
        alpha = np.asarray(rgba.getchannel("A"), dtype=np.float32) / 255.0
        upright = rgba.convert("RGB")

    rgb = _to_srgb(upright, icc)
    pixels = _to_float(rgb)
    return DecodedImage(
        pixels=pixels,
        format=fmt,
        exif=exif if isinstance(exif, bytes) else b"",
        has_alpha=alpha is not None,
        alpha=alpha,
    )


HIGHLIGHT_KNEE = 0.75
"""Linear level above which recovered RAW highlights are eased into range rather than
clipped (see `ease_highlights`)."""


def decode_raw(data: bytes) -> DecodedImage:
    """Develop a camera RAW file into upright sRGB float32 pixels.

    LibRaw demosaics the sensor data, applies the white balance the camera recorded (or
    daylight if it recorded none), and converts the camera's color response to sRGB with
    its color matrix. Highlights where some but not all color channels clipped are
    reconstructed from the others (LibRaw's "blend" mode), and the result stays linear and
    16-bit until the sRGB curve is applied here, in float.
    """
    try:
        with rawpy.imread(io.BytesIO(data)) as raw:
            balance = _white_balance(raw)
            rgb = raw.postprocess(
                output_color=rawpy.ColorSpace.sRGB,
                gamma=(1, 1),
                output_bps=16,
                no_auto_bright=True,
                use_camera_wb=True,
                highlight_mode=rawpy.HighlightMode.Blend,
            )
    except (rawpy.LibRawError, ValueError, OSError) as exc:
        raise UnsupportedImageError(
            "This file is not a photo we can open; use JPEG, PNG, HEIC, or a camera RAW file."
        ) from exc
    # LibRaw scales by the white balance so that the strongest channel still fits, which
    # leaves room above white for recovered highlights and makes everything else darker
    # by that factor. Undo it, then ease what is above white back into range.
    linear = rgb.astype(np.float32) / 65535.0 * balance
    return DecodedImage(
        pixels=linear_to_srgb(ease_highlights(linear)), format=RAW, exif=raw_exif(data)
    )


def _white_balance(raw: rawpy.RawPy) -> float:
    """How much LibRaw's white balance scaling darkened the photo (max over min gain)."""
    for gains in (raw.camera_whitebalance, raw.daylight_whitebalance):
        positive = [g for g in gains if g > 0]
        if positive:
            return float(max(positive) / min(positive))
    return 1.0


def ease_highlights(linear: Array, knee: float = HIGHLIGHT_KNEE) -> Array:
    """Keep linear values below `knee` and roll everything above it smoothly toward 1, so
    reconstructed highlights keep their detail instead of clipping to flat white."""
    above = linear > knee
    if not above.any() or float(linear.max()) <= 1.0:
        return np.clip(linear, 0.0, 1.0)
    room = 1.0 - knee
    eased = knee + room * (1.0 - np.exp(-(linear - knee) / room))
    return np.asarray(np.clip(np.where(above, eased, linear), 0.0, 1.0), dtype=np.float32)


def linear_to_srgb(linear: Array) -> Array:
    """Apply the sRGB transfer curve to linear values in [0, 1]."""
    x = np.clip(linear, 0.0, 1.0)
    out = np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)
    return np.asarray(out, dtype=np.float32)


TIFF_HEADERS = (b"II*\x00", b"MM\x00*")


def raw_exif(data: bytes) -> bytes:
    """Camera EXIF from a RAW file, as plain EXIF bytes with the orientation reset (LibRaw
    already turns the photo upright). Most RAW formats are TIFF inside, or carry a TIFF
    block of EXIF, so the first TIFF header in the file is read; empty when there is none."""
    starts = [i for i in (data.find(h, 0, 1 << 20) for h in TIFF_HEADERS) if i >= 0]
    if not starts:
        return b""
    exif = Image.Exif()
    try:
        exif.load(b"Exif\x00\x00" + data[min(starts) :])
        clean = Image.Exif()
        for tag, value in exif.items():
            if tag in (EXIF_IFD_TAG, GPS_IFD_TAG):
                continue
            if isinstance(value, (str, int, float, tuple)):
                clean[tag] = value
        for ifd in (EXIF_IFD_TAG, GPS_IFD_TAG):
            found = exif.get_ifd(ifd)
            if found:
                target = clean.get_ifd(ifd)
                for tag, value in found.items():
                    if isinstance(value, (str, int, float, tuple, bytes)):
                        target[tag] = value
        clean[EXIF_ORIENTATION] = 1
        return clean.tobytes()
    except Exception:  # Odd RAW metadata is not worth failing the upload over.
        return b""


def _to_srgb(img: Image.Image, icc: bytes | None) -> Image.Image:
    """Convert to 8- or 16-bit RGB in sRGB, applying an embedded ICC profile if present."""
    if img.mode in ("I;16", "I;16B", "I;16L", "I"):
        # 16-bit grayscale PNG: scale down to 8-bit grayscale, then RGB.
        arr = np.asarray(img, dtype=np.float32)
        maxval = 65535.0 if arr.max() > 255 else 255.0
        img = Image.fromarray(np.clip(arr / maxval * 255.0 + 0.5, 0, 255).astype(np.uint8))
    if img.mode != "RGB":
        img = img.convert("RGB")
    if icc:
        try:
            src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
            converted = ImageCms.profileToProfile(img, src, SRGB_PROFILE, outputMode="RGB")
            if converted is not None:
                img = converted
        except (ImageCms.PyCMSError, OSError):
            pass  # A broken profile is not worth failing the upload over; assume sRGB.
    return img


def _to_float(img: Image.Image) -> Array:
    arr = np.asarray(img)
    scale = 65535.0 if arr.dtype == np.uint16 else 255.0
    return (arr.astype(np.float32) / scale).copy()


def resize_long_edge(pixels: Array, long_edge: int) -> Array:
    """Downscale (never upscale) so the longer side is at most `long_edge` pixels."""
    h, w = pixels.shape[:2]
    scale = long_edge / max(h, w)
    if scale >= 1:
        return pixels
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return np.asarray(cv2.resize(pixels, size, interpolation=cv2.INTER_AREA), dtype=np.float32)


def make_proxy(pixels: Array) -> Array:
    """The downscaled working copy that interactive previews are rendered from."""
    return resize_long_edge(pixels, PREVIEW_LONG_EDGE)


def to_uint8(pixels: Array) -> npt.NDArray[np.uint8]:
    return np.asarray(np.clip(pixels * 255.0 + 0.5, 0, 255), dtype=np.uint8)


def encode_jpeg(pixels: Array, quality: int = 90, exif: bytes | None = None) -> bytes:
    img = Image.fromarray(to_uint8(pixels), mode="RGB")
    buf = io.BytesIO()
    kwargs: dict[str, object] = {"quality": quality, "optimize": True, "icc_profile": SRGB_ICC}
    if exif:
        kwargs["exif"] = exif
    img.save(buf, format="JPEG", **kwargs)
    return buf.getvalue()


def encode_png(pixels: Array, exif: bytes | None = None, alpha: Array | None = None) -> bytes:
    """Encode RGB pixels as PNG, with transparency when `alpha` (0..1) is given."""
    img = Image.fromarray(to_uint8(pixels), mode="RGB")
    if alpha is not None:
        img.putalpha(Image.fromarray(to_uint8(alpha), mode="L"))
    buf = io.BytesIO()
    kwargs: dict[str, object] = {"icc_profile": SRGB_ICC}
    if exif:
        kwargs["exif"] = exif
    img.save(buf, format="PNG", **kwargs)
    return buf.getvalue()


def encode_gray_png(values: Array) -> bytes:
    """Encode a single-channel 0..1 array as an 8-bit grayscale PNG."""
    img = Image.fromarray(to_uint8(values), mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
