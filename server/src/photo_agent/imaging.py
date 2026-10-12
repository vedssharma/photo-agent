"""Image ingest and encoding: decode uploads into the working color space and back.

The working representation is an RGB float32 NumPy array of shape (height, width, 3)
with values in [0, 1], encoded with the sRGB tone curve in the photo's working color space:
sRGB for ordinary photos, Rec.2020 for RAW and wide-gamut ones (see `photo_agent.color`).
Pixels stay float from decode to encode, so 16-bit and 10-bit sources keep every bit.
Every decoder output is rotated upright per its EXIF orientation, so the rest of the
pipeline never thinks about orientation again. Encoders take sRGB pixels; `to_srgb` is the
output transform for everything else.

Camera RAW files are developed by LibRaw (through `rawpy`): demosaiced, white balanced as
shot, converted from the camera's own color response with its color matrix, and with
clipped highlights reconstructed rather than cut off.
"""

from __future__ import annotations

import io
import struct
import zlib
from collections.abc import Callable
from dataclasses import dataclass, field

import cv2
import numpy as np
import numpy.typing as npt
import pillow_heif
import rawpy
import tifffile
from PIL import Image, ImageCms, ImageOps, UnidentifiedImageError

from photo_agent import color, icc

pillow_heif.register_heif_opener()

Array = npt.NDArray[np.float32]

SUPPORTED_FORMATS = {"JPEG", "PNG", "HEIF", "TIFF"}
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
    """Upright RGB float32 pixels in [0, 1], in `space` (encoded with the sRGB curve)."""
    format: str
    """Source format: JPEG, PNG, HEIF, TIFF, or RAW."""
    exif: bytes = b""
    """Raw EXIF from the source (orientation not yet reset), or empty."""
    has_alpha: bool = False
    alpha: Array | None = field(default=None, repr=False)
    """Upright alpha channel in [0, 1] when the source had one."""
    space: color.WorkingSpace = "srgb"
    """The working color space the pixels are in (see `photo_agent.color`)."""

    @property
    def width(self) -> int:
        return int(self.pixels.shape[1])

    @property
    def height(self) -> int:
        return int(self.pixels.shape[0])


def decode(data: bytes) -> DecodedImage:
    """Decode JPEG, PNG, HEIC, TIFF, or camera RAW bytes into upright float32 pixels in
    their working color space, at their full bit depth."""
    if data[:4] in TIFF_HEADERS:
        # Many RAW formats are TIFF underneath. LibRaw turns down ordinary TIFFs, so it
        # gets the first look.
        try:
            return decode_raw(data)
        except UnsupportedImageError:
            return decode_tiff(data)
    try:
        img = Image.open(io.BytesIO(data))
        fmt = FORMAT_ALIASES.get(img.format or "", img.format)
    except (UnidentifiedImageError, OSError) as exc:
        try:
            return decode_raw(data)
        except UnsupportedImageError:
            raise UnsupportedImageError("Could not read this file as an image.") from exc
    if fmt not in SUPPORTED_FORMATS:
        raise UnsupportedImageError(
            f"{img.format or 'This'} files are not supported yet; "
            "use JPEG, PNG, HEIC, TIFF, or a camera RAW file."
        )
    try:
        img.load()
    except OSError as exc:
        raise UnsupportedImageError("Could not read this file as an image.") from exc
    exif = img.info.get("exif", b"")
    exif = exif if isinstance(exif, bytes) else b""
    embedded = img.info.get("icc_profile") or None
    profile = icc.parse(embedded) if embedded else None

    deep = _high_bit_depth(fmt, data, img)
    if deep is not None:
        # 16-bit PNGs and 10- or 12-bit HEICs: keep every bit rather than Pillow's 8.
        values, deep_alpha = _split_alpha(_orient(deep, _orientation(img)))
        values, space = apply_profile(values, profile)
        return DecodedImage(
            pixels=values,
            format=fmt,
            exif=exif,
            has_alpha=deep_alpha is not None,
            alpha=deep_alpha,
            space=space,
        )

    upright = ImageOps.exif_transpose(img)
    alpha: Array | None = None
    if upright.mode in ("RGBA", "LA") or (upright.mode == "P" and "transparency" in upright.info):
        rgba = upright.convert("RGBA")
        alpha = np.asarray(rgba.getchannel("A"), dtype=np.float32) / 255.0
        upright = rgba.convert("RGB")

    if profile is None:
        # No profile, or one that is not a simple matrix profile: LittleCMS takes it to
        # sRGB (in 8 bits, which is all these sources have).
        pixels, space = _to_float(_to_srgb(upright, embedded)), "srgb"
    else:
        pixels, space = apply_profile(_to_float(_to_srgb(upright, None)), profile)
    return DecodedImage(
        pixels=pixels,
        format=fmt,
        exif=exif,
        has_alpha=alpha is not None,
        alpha=alpha,
        space=space,
    )


def apply_profile(
    pixels: Array, profile: icc.MatrixProfile | None
) -> tuple[Array, color.WorkingSpace]:
    """Pixels encoded per an embedded profile, converted in float into the working space
    that holds all their colors: sRGB when the profile fits inside it, else the wide one.
    No profile means sRGB, as browsers assume."""
    if profile is None or profile.is_srgb():
        return pixels, "srgb"
    space: color.WorkingSpace = "srgb" if profile.fits_in("srgb") else color.WIDE
    to_space = np.linalg.inv(color.to_xyz_matrix(space)) @ profile.to_xyz
    linear = profile.to_linear(pixels) @ to_space.T.astype(np.float32)
    return color.encode_curve(np.clip(linear, 0.0, 1.0)), space


def _high_bit_depth(fmt: str, data: bytes, img: Image.Image) -> Array | None:
    """The pixels as float in [0, 1] (RGB or RGBA) when the file holds more than 8 bits
    per channel, which Pillow would round away; None otherwise."""
    if fmt == "PNG" and len(data) > 24 and data[24] == 16:
        raw = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
        if raw is None:
            return None
        if raw.ndim == 2:
            raw = raw[..., None]
        if raw.shape[2] == 2:  # Gray and alpha.
            raw = raw[..., [0, 0, 0, 1]]
        elif raw.shape[2] == 1:
            raw = np.repeat(raw, 3, axis=2)
        else:
            raw = raw[..., [2, 1, 0, 3][: raw.shape[2]]]  # OpenCV is BGR(A).
        return np.asarray(raw, dtype=np.float32) / 65535.0
    if fmt == "HEIF" and int(img.info.get("bit_depth", 8)) > 8:
        heif = pillow_heif.open_heif(io.BytesIO(data), convert_hdr_to_8bit=False)
        return np.asarray(heif, dtype=np.float32) / 65535.0
    return None


def _split_alpha(values: Array) -> tuple[Array, Array | None]:
    if values.shape[2] == 4:
        return np.ascontiguousarray(values[..., :3]), np.ascontiguousarray(values[..., 3])
    return np.ascontiguousarray(values[..., :3]), None


def _orientation(img: Image.Image) -> int:
    try:
        return int(img.getexif().get(EXIF_ORIENTATION, 1))
    except (ValueError, TypeError, OSError):
        return 1


def _orient(pixels: Array, orientation: int) -> Array:
    """Turn pixels stored with an EXIF/TIFF orientation upright (as ImageOps.exif_transpose
    does for Pillow images)."""
    flips: dict[int, Callable[[Array], Array]] = {
        2: lambda a: a[:, ::-1],
        3: lambda a: a[::-1, ::-1],
        4: lambda a: a[::-1],
        5: lambda a: np.swapaxes(a, 0, 1),
        6: lambda a: np.rot90(a, -1),
        7: lambda a: np.rot90(a, 1)[::-1],
        8: lambda a: np.rot90(a, 1),
    }
    turn = flips.get(orientation)
    return np.ascontiguousarray(turn(pixels)) if turn else pixels


def decode_tiff(data: bytes) -> DecodedImage:
    """An ordinary (not RAW) TIFF: 8- or 16-bit, or 32-bit float, RGB or gray, with or
    without alpha. Float TIFFs hold linear light (as HDR merges and 32-bit editors write
    them), so they are given the sRGB curve, with anything above white eased into range."""
    try:
        with tifffile.TiffFile(io.BytesIO(data)) as tif:
            page = tif.pages.first
            raw = page.asarray()
            orientation = int(page.tags.valueof(274, 1) or 1)
            embedded = page.tags.valueof(34675)
            extra = len(page.extrasamples or ())
            planar = page.planarconfig == tifffile.PLANARCONFIG.SEPARATE
    except (tifffile.TiffFileError, ValueError, OSError, IndexError) as exc:
        raise UnsupportedImageError("Could not read this TIFF file.") from exc
    arr = np.asarray(raw)
    if planar and arr.ndim == 3:
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.ndim != 3:
        raise UnsupportedImageError("This TIFF has more than one image in it.")
    if np.issubdtype(arr.dtype, np.floating):
        values = arr.astype(np.float32)
        color = values[..., : arr.shape[2] - extra] if extra else values
        values[..., : color.shape[2]] = linear_to_srgb(ease_highlights(np.maximum(color, 0)))
    elif arr.dtype == np.uint8:
        values = arr.astype(np.float32) / 255.0
    elif arr.dtype == np.uint16:
        values = arr.astype(np.float32) / 65535.0
    else:
        raise UnsupportedImageError("This TIFF's pixel format is not supported.")
    colors = values.shape[2] - extra
    if colors == 1:
        values = np.concatenate([np.repeat(values[..., :1], 3, axis=2), values[..., 1:]], 2)
    elif colors != 3:
        raise UnsupportedImageError("Only RGB and grayscale TIFFs are supported.")
    values = np.asarray(values[..., :4] if extra else values[..., :3], dtype=np.float32)
    pixels, alpha = _split_alpha(_orient(values, orientation))
    profile = icc.parse(bytes(embedded)) if isinstance(embedded, (bytes, np.ndarray)) else None
    pixels, space = apply_profile(pixels, profile)
    return DecodedImage(
        pixels=pixels,
        format="TIFF",
        exif=raw_exif(data),
        has_alpha=alpha is not None,
        alpha=alpha,
        space=space,
    )


HIGHLIGHT_KNEE = 0.75
"""Linear level above which recovered RAW highlights are eased into range rather than
clipped (see `ease_highlights`)."""


def decode_raw(data: bytes) -> DecodedImage:
    """Develop a camera RAW file into upright float32 pixels in the wide working space.

    LibRaw demosaics the sensor data, applies the white balance the camera recorded (or
    daylight if it recorded none), and converts the camera's color response to linear
    Rec.2020 with its color matrix, so no color the sensor saw is clipped. Highlights where
    some but not all color channels clipped are reconstructed from the others (LibRaw's
    "blend" mode), and the result stays linear and 16-bit until the tone curve is applied
    here, in float.
    """
    try:
        with rawpy.imread(io.BytesIO(data)) as raw:
            balance = _white_balance(raw)
            rgb = raw.postprocess(
                output_color=rawpy.ColorSpace.Rec2020,
                gamma=(1, 1),
                output_bps=16,
                no_auto_bright=True,
                use_camera_wb=True,
                highlight_mode=rawpy.HighlightMode.Blend,
            )
    except (rawpy.LibRawError, ValueError, OSError) as exc:
        raise UnsupportedImageError(
            "This file is not a photo we can open; use JPEG, PNG, HEIC, TIFF, or a camera RAW file."
        ) from exc
    # LibRaw scales by the white balance so that the strongest channel still fits, which
    # leaves room above white for recovered highlights and makes everything else darker
    # by that factor. Undo it, then ease what is above white back into range.
    linear = rgb.astype(np.float32) / 65535.0 * balance
    return DecodedImage(
        pixels=linear_to_srgb(ease_highlights(linear)),
        format=RAW,
        exif=raw_exif(data),
        space=color.WIDE,
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
    """Apply the sRGB tone curve to linear values in [0, 1]."""
    return color.encode_curve(np.clip(linear, 0.0, 1.0))


TIFF_HEADERS = (b"II*\x00", b"MM\x00*")

CAMERA_TAGS = (270, 271, 272, 305, 306, 315, 33432)
"""Main-IFD EXIF tags worth carrying over: description, make, model, software, date,
artist, and copyright. The rest of a TIFF's main IFD describes its own pixel layout."""


def raw_exif(data: bytes) -> bytes:
    """Camera EXIF from a RAW or TIFF file, as plain EXIF bytes. Most RAW formats are TIFF inside,
    or carry a TIFF block of EXIF, so the first TIFF header in the file is read; empty
    when there is none."""
    starts = [i for i in (data.find(h, 0, 1 << 20) for h in TIFF_HEADERS) if i >= 0]
    if not starts:
        return b""
    exif = Image.Exif()
    try:
        exif.load(b"Exif\x00\x00" + data[min(starts) :])
    except Exception:  # Odd RAW metadata is not worth failing the upload over.
        return b""
    return camera_exif(exif)


def camera_exif(exif: Image.Exif) -> bytes:
    """The camera and shot details (and location) from a TIFF-style IFD, as EXIF bytes
    with the orientation reset: the decoder has already turned the photo upright."""
    try:
        clean = Image.Exif()
        for tag in CAMERA_TAGS:
            value = exif.get(tag)
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
    except Exception:  # Odd metadata is not worth failing the upload over.
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


def to_srgb(pixels: Array, space: color.Space) -> Array:
    """Pixels in a working space as sRGB, for screens, Claude, and image models; colors
    outside sRGB are mapped onto its edge."""
    return color.convert(pixels, space, "srgb")


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


def encode_jpeg(
    pixels: Array, quality: int = 90, exif: bytes | None = None, profile: bytes = SRGB_ICC
) -> bytes:
    """Encode RGB pixels as JPEG, tagged with `profile` (the ICC profile of the space the
    pixels are in; sRGB unless said otherwise)."""
    img = Image.fromarray(to_uint8(pixels), mode="RGB")
    buf = io.BytesIO()
    kwargs: dict[str, object] = {"quality": quality, "optimize": True, "icc_profile": profile}
    if exif:
        kwargs["exif"] = exif
    img.save(buf, format="JPEG", **kwargs)
    return buf.getvalue()


def encode_png(
    pixels: Array,
    exif: bytes | None = None,
    alpha: Array | None = None,
    profile: bytes = SRGB_ICC,
    bits: int = 8,
) -> bytes:
    """Encode RGB pixels as PNG, with transparency when `alpha` (0..1) is given, tagged
    with `profile`; 8 or 16 bits per channel."""
    if bits == 16:
        return _encode_png16(pixels, exif, alpha, profile)
    img = Image.fromarray(to_uint8(pixels), mode="RGB")
    if alpha is not None:
        img.putalpha(Image.fromarray(to_uint8(alpha), mode="L"))
    buf = io.BytesIO()
    kwargs: dict[str, object] = {"icc_profile": profile}
    if exif:
        kwargs["exif"] = exif
    img.save(buf, format="PNG", **kwargs)
    return buf.getvalue()


def to_uint16(pixels: Array) -> npt.NDArray[np.uint16]:
    return np.asarray(np.clip(pixels * 65535.0 + 0.5, 0, 65535), dtype=np.uint16)


def _with_alpha(pixels: Array, alpha: Array | None) -> Array:
    if alpha is None:
        return pixels
    return np.concatenate([pixels, alpha[..., None]], axis=2)


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    crc = zlib.crc32(kind + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", crc)


def _encode_png16(pixels: Array, exif: bytes | None, alpha: Array | None, profile: bytes) -> bytes:
    """A 16-bit PNG (which Pillow cannot write): OpenCV encodes the pixels, and the color
    profile and EXIF chunks are added after the header."""
    values = to_uint16(_with_alpha(pixels, alpha))
    order = [2, 1, 0, 3][: values.shape[2]]  # OpenCV wants BGR(A).
    ok, encoded = cv2.imencode(".png", np.ascontiguousarray(values[..., order]))
    if not ok:
        raise ValueError("Could not encode the PNG.")
    data = encoded.tobytes()
    header_end = 8 + 25  # Signature, then the IHDR chunk.
    extra = _png_chunk(b"iCCP", b"ICC profile\0\0" + zlib.compress(profile))
    if exif:
        extra += _png_chunk(b"eXIf", exif[6:] if exif.startswith(b"Exif\0\0") else exif)
    return data[:header_end] + extra + data[header_end:]


CAMERA_TIFF_TAGS = (271, 272, 305, 315, 33432)
"""Make, model, software, artist, and copyright: the EXIF a TIFF keeps in its main IFD."""


def encode_tiff(
    pixels: Array,
    exif: bytes | None = None,
    alpha: Array | None = None,
    profile: bytes = SRGB_ICC,
    bits: int = 16,
) -> bytes:
    """Encode RGB pixels as a TIFF (zip-compressed), 8 or 16 bits per channel, tagged with
    `profile`. The camera, date, and authorship from `exif` go in the TIFF's own tags."""
    values = _with_alpha(pixels, alpha)
    data = to_uint16(values) if bits == 16 else to_uint8(values)
    tags: list[tuple[int, str, int, object, bool]] = [(34675, "B", len(profile), profile, True)]
    if exif:
        found = Image.Exif()
        try:
            found.load(exif)
            sub = dict(found.get_ifd(EXIF_IFD_TAG))
        except Exception:  # No camera details, then.
            found, sub = Image.Exif(), {}
        for tag in CAMERA_TIFF_TAGS:
            value = found.get(tag)
            if isinstance(value, str) and value.strip("\0 "):
                tags.append((tag, "s", 0, value.strip("\0 "), True))
        when = sub.get(0x9003) or found.get(306)  # When it was taken, else last changed.
        if isinstance(when, str) and when.strip("\0 "):
            tags.append((306, "s", 0, when.strip("\0 "), True))
    buf = io.BytesIO()
    tifffile.imwrite(
        buf,
        data,
        photometric="rgb",
        extrasamples=["unassalpha"] if alpha is not None else None,
        compression="zlib",
        extratags=tags,
        metadata=None,
    )
    return buf.getvalue()


def encode_pillow(
    pixels: Array,
    fmt: str,
    quality: int,
    exif: bytes | None = None,
    alpha: Array | None = None,
    profile: bytes = SRGB_ICC,
) -> bytes:
    """Encode 8-bit RGB(A) pixels in a format Pillow writes with a quality setting (AVIF,
    WebP), tagged with `profile` and carrying `exif`."""
    img = Image.fromarray(to_uint8(pixels), mode="RGB")
    if alpha is not None:
        img.putalpha(Image.fromarray(to_uint8(alpha), mode="L"))
    kwargs: dict[str, object] = {"quality": quality, "icc_profile": profile}
    if exif:
        kwargs["exif"] = exif
    buf = io.BytesIO()
    img.save(buf, format=fmt, **kwargs)
    return buf.getvalue()


def encode_gray_png(values: Array) -> bytes:
    """Encode a single-channel 0..1 array as an 8-bit grayscale PNG."""
    img = Image.fromarray(to_uint8(values), mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
