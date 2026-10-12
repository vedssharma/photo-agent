"""Color spaces: the working spaces photos are edited in, and the output spaces they leave in.

Every space here is RGB with D65 white, encoded with the sRGB tone curve, so values in
[0, 1] mean the same lightness whatever the space and every operation tuned on sRGB
values behaves the same way. Linearizing (the inverse curve) gives linear light in that
space's primaries, which is what exposure and other light math run on.

Photos whose colors fit in sRGB (most JPEGs, PNGs, and HEICs) are edited in sRGB, as
always. Camera RAW files, and photos tagged with a wider profile, are edited in
**Rec.2020**, which holds every color a camera or a wide-gamut display can show, so
saturated colors survive editing instead of being clipped on the way in. Whatever leaves
the app (previews, Claude's view, exports) goes through `convert`, which maps colors that
do not fit the target space onto its edge rather than clipping each channel.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float32]

Space = Literal["srgb", "display-p3", "rec2020"]
"""A color space by name: its primaries, D65 white, and the sRGB tone curve."""

WorkingSpace = Literal["srgb", "rec2020"]
"""The spaces photos are edited in."""

WIDE: WorkingSpace = "rec2020"
"""The working space for RAW and wide-gamut photos."""

SPACE_NAMES: dict[Space, str] = {
    "srgb": "sRGB",
    "display-p3": "Display P3",
    "rec2020": "Rec.2020",
}

PRIMARIES: dict[Space, tuple[tuple[float, float], ...]] = {
    "srgb": ((0.640, 0.330), (0.300, 0.600), (0.150, 0.060)),
    "display-p3": ((0.680, 0.320), (0.265, 0.690), (0.150, 0.060)),
    "rec2020": ((0.708, 0.292), (0.170, 0.797), (0.131, 0.046)),
}
"""Chromaticities (x, y) of each space's red, green, and blue."""

D65 = (0.3127, 0.3290)


def _xyz(x: float, y: float) -> npt.NDArray[np.float64]:
    return np.array([x / y, 1.0, (1 - x - y) / y])


def to_xyz_matrix(space: Space) -> npt.NDArray[np.float64]:
    """The matrix taking linear RGB in `space` to CIE XYZ (D65, Y of white = 1)."""
    columns = np.stack([_xyz(*xy) for xy in PRIMARIES[space]], axis=1)
    scale = np.linalg.solve(columns, _xyz(*D65))
    return columns * scale


def matrix(src: Space, dst: Space) -> npt.NDArray[np.float64]:
    """The matrix taking linear RGB in `src` to linear RGB in `dst`."""
    return np.linalg.inv(to_xyz_matrix(dst)) @ to_xyz_matrix(src)


def luminance_weights(space: Space) -> npt.NDArray[np.float32]:
    """How much each linear channel of `space` contributes to luminance (Y)."""
    return np.asarray(to_xyz_matrix(space)[1], dtype=np.float32)


def decode_curve(x: Array) -> Array:
    """The sRGB tone curve undone: encoded values to linear light. Negative values (colors
    outside the space) mirror the curve, so conversions round-trip."""
    a = np.abs(x)
    linear = np.where(a <= 0.04045, a / 12.92, np.power((a + 0.055) / 1.055, 2.4))
    return np.asarray(np.sign(x) * linear, dtype=np.float32)


def encode_curve(x: Array) -> Array:
    """Linear light to values encoded with the sRGB tone curve (mirrored below zero)."""
    a = np.abs(x)
    encoded = np.where(a <= 0.0031308, a * 12.92, 1.055 * np.power(a, 1 / 2.4) - 0.055)
    return np.asarray(np.sign(x) * encoded, dtype=np.float32)


def convert_linear(linear: Array, src: Space, dst: Space) -> Array:
    """Linear RGB in `src` as linear RGB in `dst`, unclipped."""
    if src == dst:
        return linear
    m = matrix(src, dst).astype(np.float32)
    return np.asarray(linear @ m.T, dtype=np.float32)


def fit_gamut(linear: Array, space: Space) -> Array:
    """Bring colors outside `space` (a channel below zero) onto its edge by desaturating
    them toward their own luminance, which keeps their hue and brightness, instead of
    clipping channels one by one, which shifts hue. Then clip to [0, 1]."""
    weights = luminance_weights(space)
    y = np.clip(linear @ weights, 0.0, None)[..., None]
    low = linear.min(axis=-1, keepdims=True)
    # The share of the color's distance from gray that still fits.
    keep = np.where(low < 0, y / np.maximum(y - low, 1e-9), 1.0)
    fitted = y + (linear - y) * np.clip(keep, 0.0, 1.0)
    return np.asarray(np.clip(fitted, 0.0, 1.0), dtype=np.float32)


def convert(pixels: Array, src: Space, dst: Space) -> Array:
    """Encoded pixels in `src` as encoded pixels in `dst`, with colors that do not fit the
    target mapped onto its edge (see `fit_gamut`)."""
    if src == dst:
        return pixels
    linear = convert_linear(decode_curve(pixels), src, dst)
    return encode_curve(fit_gamut(linear, dst))


def out_of_gamut(pixels: Array, src: Space, dst: Space, tolerance: float = 1 / 255) -> Array:
    """Where colors in `src` do not fit `dst`: a 0/1 float map."""
    if src == dst:
        return np.zeros(pixels.shape[:2], dtype=np.float32)
    linear = convert_linear(decode_curve(np.clip(pixels, 0.0, 1.0)), src, dst)
    encoded = encode_curve(linear)
    outside = (encoded < -tolerance) | (encoded > 1 + tolerance)
    return np.asarray(outside.any(axis=-1), dtype=np.float32)


def contains(outer: Space, inner: Space) -> bool:
    """Whether every color of `inner` fits in `outer`."""
    order: list[Space] = ["srgb", "display-p3", "rec2020"]
    return order.index(outer) >= order.index(inner)
