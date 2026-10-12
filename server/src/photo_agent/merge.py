"""Combine several photos into one: HDR from exposure brackets, and panoramas.

The result becomes a new photo in the library, saved as a 16-bit TIFF in the sources'
working color space with its ICC profile, so nothing is lost before editing starts.
The sources stay as they were.
"""

from __future__ import annotations

import io
from collections.abc import Sequence
from typing import Literal

import cv2
import numpy as np
import numpy.typing as npt
import tifffile
from PIL import Image

from photo_agent import color, icc, imaging

Array = npt.NDArray[np.float32]
MergeKind = Literal["hdr", "panorama"]

MAX_SOURCES = 9


class MergeError(ValueError):
    """The photos cannot be merged; the message says why, in plain words."""


def common_space(images: Sequence[imaging.DecodedImage]) -> color.WorkingSpace:
    """The working space that holds every source's colors."""
    return color.WIDE if any(img.space != "srgb" for img in images) else "srgb"


def in_space(img: imaging.DecodedImage, space: color.WorkingSpace) -> Array:
    return color.convert(img.pixels, img.space, space)


# HDR


def merge_hdr(images: Sequence[imaging.DecodedImage]) -> tuple[Array, color.WorkingSpace]:
    """Blend a bracket of exposures into one photo with detail in both the shadows and the
    highlights (Mertens exposure fusion: each pixel is taken mostly from the shots where
    it is well exposed, saturated, and in focus), after aligning hand-held frames."""
    if len(images) < 2:
        raise MergeError("Choose at least two exposures of the same scene to merge.")
    sizes = {(img.width, img.height) for img in images}
    if len(sizes) > 1:
        raise MergeError("The exposures are different sizes, so they are not one bracket.")
    space = common_space(images)
    frames = [in_space(img, space) for img in images]
    frames = align(frames)
    fused = cv2.createMergeMertens().process([f * 255.0 for f in frames])
    return np.asarray(np.clip(fused, 0.0, 1.0), dtype=np.float32), space


MAX_SHIFT_BITS = 6
"""Hand-held brackets may be up to 2**6 = 64 pixels apart."""


def align(frames: Sequence[Array]) -> list[Array]:
    """Shift each frame onto the middle one. Median threshold bitmaps compare frames by
    structure, not brightness, so differently exposed shots line up."""
    gray = [cv2.cvtColor(imaging.to_uint8(np.clip(f, 0, 1)), cv2.COLOR_RGB2GRAY) for f in frames]
    reference = len(frames) // 2
    mtb = cv2.createAlignMTB(MAX_SHIFT_BITS)
    out: list[Array] = []
    for frame, g in zip(frames, gray, strict=True):
        dx, dy = mtb.calculateShift(gray[reference], g)
        if dx == 0 and dy == 0:
            out.append(frame)
            continue
        shift = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
        h, w = frame.shape[:2]
        moved = cv2.warpAffine(frame, shift, (w, h), borderMode=cv2.BORDER_REPLICATE)
        out.append(np.asarray(moved, dtype=np.float32))
    return out


# Panoramas


def stitch_panorama(images: Sequence[imaging.DecodedImage]) -> tuple[Array, color.WorkingSpace]:
    """Stitch overlapping photos into one wide view: features are matched between them,
    the camera's turn is estimated, and the photos are warped and blended with seams
    through low-detail areas. The ragged edges are cropped away."""
    if len(images) < 2:
        raise MergeError("Choose at least two overlapping photos to stitch.")
    space = common_space(images)
    frames = [in_space(img, space) for img in images]
    # OpenCV's stitcher only blends 8-bit images (its exposure compensation insists), so
    # panoramas are stitched at 8 bits per channel, in the sources' working space so wide
    # colors survive.
    eight = [cv2.cvtColor(imaging.to_uint8(np.clip(f, 0, 1)), cv2.COLOR_RGB2BGR) for f in frames]
    largest = max(f.shape[0] * f.shape[1] for f in frames)
    total = sum(f.shape[0] * f.shape[1] for f in frames)
    for mode in (cv2.Stitcher_PANORAMA, cv2.Stitcher_SCANS):
        stitcher = cv2.Stitcher.create(mode)
        status = stitcher.estimateTransform(eight)
        if status != cv2.Stitcher_OK or len(stitcher.component()) < len(frames):
            continue
        pano = _compose(stitcher, eight)
        if pano is None:
            continue
        # A plausible panorama is bigger than any one photo and smaller than all of them
        # side by side; anything else is a failed guess at the geometry.
        area = pano.shape[0] * pano.shape[1]
        if largest * 0.9 <= area <= total * 1.2:
            return crop_to_content(pano), space
    raise MergeError(
        "These photos could not be stitched: they need to overlap by about a third, "
        "taken from one spot while turning."
    )


def _compose(stitcher: cv2.Stitcher, eight: Sequence[cv2.typing.MatLike]) -> Array | None:
    """The panorama, as float RGB, once the stitcher has worked out the geometry."""
    status, pano = stitcher.composePanorama(list(eight))
    if status != cv2.Stitcher_OK or pano is None:
        return None
    rgb = cv2.cvtColor(np.asarray(pano), cv2.COLOR_BGR2RGB)
    return np.asarray(rgb, dtype=np.float32) / 255.0


def crop_to_content(pano: Array) -> Array:
    """The largest upright rectangle without the empty (black) areas warping leaves at
    the edges, found by shrinking the bounding box from whichever side has the most
    empty pixels."""
    filled = cv2.erode((pano.max(axis=2) > 1e-4).astype(np.uint8), np.ones((3, 3), np.uint8))
    ys, xs = np.nonzero(filled)
    if ys.size == 0:
        return pano
    top, bottom, left, right = ys.min(), ys.max(), xs.min(), xs.max()
    while top < bottom and left < right:
        box = filled[top : bottom + 1, left : right + 1]
        if box.all():
            break
        empty = {
            "top": int((box[0] == 0).sum()) / box.shape[1],
            "bottom": int((box[-1] == 0).sum()) / box.shape[1],
            "left": int((box[:, 0] == 0).sum()) / box.shape[0],
            "right": int((box[:, -1] == 0).sum()) / box.shape[0],
        }
        side = max(empty, key=lambda k: empty[k])
        if side == "top":
            top += 1
        elif side == "bottom":
            bottom -= 1
        elif side == "left":
            left += 1
        else:
            right -= 1
    return np.ascontiguousarray(pano[top : bottom + 1, left : right + 1])


# Saving


def encode_tiff(pixels: Array, space: color.WorkingSpace, like: imaging.DecodedImage) -> bytes:
    """A 16-bit TIFF of the pixels, tagged with their color space's profile, and with the
    camera and date of `like` (the first source) so the library files it alongside them."""
    data = np.round(np.clip(pixels, 0, 1) * 65535).astype(np.uint16)
    profile = icc.profile(space)
    tags: list[tuple[int, str, int, object, bool]] = [(34675, "B", len(profile), profile, True)]
    exif = Image.Exif()
    if like.exif:
        try:
            exif.load(like.exif)
        except Exception:  # No camera details, then.
            exif = Image.Exif()
    sub = _exif_ifd(exif)
    when = sub.get(0x9003) or exif.get(306)
    for tag, value in ((271, exif.get(271)), (272, exif.get(272)), (306, when)):
        if isinstance(value, str) and value.strip("\0 "):
            tags.append((tag, "s", 0, value.strip("\0 "), True))
    buf = io.BytesIO()
    tifffile.imwrite(
        buf, data, photometric="rgb", compression="zlib", extratags=tags, metadata=None
    )
    return buf.getvalue()


def _exif_ifd(exif: Image.Exif) -> dict[int, object]:
    try:
        return dict(exif.get_ifd(0x8769))
    except Exception:
        return {}


def merge(kind: MergeKind, sources: Sequence[bytes]) -> tuple[bytes, color.WorkingSpace]:
    """Merge the source files' photos; the result as TIFF bytes. Raises MergeError."""
    if len(sources) > MAX_SOURCES:
        raise MergeError(f"Merge at most {MAX_SOURCES} photos at once.")
    images = [imaging.decode(data) for data in sources]
    pixels, space = merge_hdr(images) if kind == "hdr" else stitch_panorama(images)
    return encode_tiff(pixels, space, images[0]), space
