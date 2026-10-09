"""Render engine: apply an operation list to pixels.

Operations run in order on RGB float32 arrays (sRGB-encoded, nominally 0..1). Values may
leave that range between steps, so a later step can pull back what an earlier one pushed
out; the result is clipped once at the end.

Every spatial parameter is expressed relative to the original image, and `RenderContext`
says how this render's pixels relate to it, so a preview proxy and a full-resolution export
of the same graph look the same apart from resolution.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Protocol, cast

import cv2
import numpy as np
import numpy.typing as npt

from photo_agent import operations as ops
from photo_agent.compositing import harmonize
from photo_agent.generative import cache_identity, job_params, task_for
from photo_agent.imaging import Array
from photo_agent.layers import BlendMode, EditState, Layer
from photo_agent.masks import Mask, SemanticMask, render_mask
from photo_agent.vision.classical import guided_filter

LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


class Vision(Protocol):
    """Runs AI models for a document (see `photo_agent.vision.selection`)."""

    def mask(
        self, mask: SemanticMask, framing: Sequence[ops.OpBase], shape: tuple[int, int]
    ) -> Array:
        """A semantic mask's selection, at `shape` (height, width) of the framed photo."""
        ...

    def fill(self, image: Array, hole: npt.NDArray[np.bool_], key: str) -> Array:
        """`image` with the `hole` filled in by inpainting; cached under `key`."""
        ...

    def generate(
        self,
        task: str,
        image: Array,
        hole: npt.NDArray[np.bool_] | None,
        params: dict[str, Any],
        key: str,
        edge: int = ...,
    ) -> Array:
        """`image` with the region around `hole` (or all of it) generated anew by `task`;
        cached under `key` at a fixed working size (long edge `edge`), whatever the render
        size."""
        ...


@dataclass(frozen=True)
class RenderContext:
    scale: float = 1.0
    """Pixels in this render per pixel of the original (1.0 for export, <1 for previews)."""
    source_aspect: float = 1.0
    """Width / height of the original photo, for the "original" crop ratio."""
    vision: Vision | None = field(default=None, compare=False)
    """Runs the models behind semantic masks; without it they cannot render."""
    framing: tuple[ops.OpBase, ...] = field(default=(), compare=False)
    """The framing of the state being rendered, which semantic masks are found in."""

    def mask(self, mask: Mask, x: Array) -> Array:
        """Render a layer mask for the pixels `x`."""
        vision = self.vision
        if vision is None:
            return render_mask(mask, x)
        return render_mask(mask, x, lambda m, shape: vision.mask(m, self.framing, shape))


def render(pixels: Array, operations: Sequence[ops.OpBase], ctx: RenderContext) -> Array:
    """Apply a flat operation list in order."""
    out = apply_operations(pixels.astype(np.float32, copy=True), operations, ctx)
    return np.clip(out, 0.0, 1.0, out=out)


def render_state(pixels: Array, state: EditState, ctx: RenderContext) -> Array:
    """Apply the framing, then the removal layers, then blend in each visible adjustment
    layer from the bottom up. A cutout's background shows as its color, or as a
    checkerboard where it is transparent."""
    rgb, alpha = render_cutout(pixels, state, ctx)
    if alpha is None or state.cutout is None:
        return rgb
    backdrop = _backdrop(rgb.shape[:2], state.cutout.background)
    return cast(Array, backdrop + (rgb - backdrop) * alpha[..., None])


def render_cutout(
    pixels: Array, state: EditState, ctx: RenderContext
) -> tuple[Array, Array | None]:
    """The rendered photo and, when a visible cutout is set, its alpha (1 keeps a pixel)."""
    ctx = replace(ctx, framing=tuple(state.framing))
    out = apply_operations(pixels.astype(np.float32, copy=True), state.framing, ctx)
    out = _apply_content(out, state.layers, ctx)
    for layer in state.layers:
        out = _apply_layer(out, layer, ctx)
    out = np.clip(out, 0.0, 1.0, out=out)
    if state.cutout is None or not state.cutout.visible:
        return out, None
    return out, np.clip(ctx.mask(state.cutout.mask, out), 0.0, 1.0)


CHECKER = (0.8, 0.6)
"""Light and dark squares behind a transparent background in previews."""


def _backdrop(shape: tuple[int, int], color: str | None) -> Array:
    h, w = shape
    if color is not None:
        rgb = [int(color[i : i + 2], 16) / 255 for i in (1, 3, 5)]
        return np.broadcast_to(np.array(rgb, np.float32), (h, w, 3)).astype(np.float32)
    cell = max(4, round(max(h, w) * 0.015))
    ys, xs = np.mgrid[0:h, 0:w]
    dark = ((ys // cell + xs // cell) % 2).astype(bool)
    gray = np.where(dark, CHECKER[1], CHECKER[0]).astype(np.float32)
    return np.repeat(gray[..., None], 3, axis=2)


def _apply_layer(x: Array, layer: Layer, ctx: RenderContext) -> Array:
    if not layer.visible or layer.opacity <= 0 or not layer.operations or layer.is_content:
        return x
    adjusted = blend(x, apply_operations(x, layer.operations, ctx), layer.blend_mode)
    weight: Array | float = layer.opacity / 100
    if layer.mask is not None:
        weight = ctx.mask(layer.mask, x)[..., None] * weight
    return cast(Array, x + (adjusted - x) * weight)


CUTOUT_ID = "cutout"
"""Stands for the cutout where a layer id is expected (layer ids start with "L")."""

REMOVAL_THRESHOLD = 0.35
"""Where a removal layer's mask is at least this strong, the photo is filled in."""


def _apply_content(x: Array, layers: Sequence[Layer], ctx: RenderContext) -> Array:
    """Apply each visible content layer (removals and generative edits), in stack order.

    Content layers come before every adjustment, whatever their place in the stack, so new
    or filled-in areas take each adjustment just like the photo around them. Each result is
    found from the photo as earlier content layers left it, and is cached by everything it
    depends on.
    """
    chain: list[object] = [[op.model_dump(exclude={"id"}) for op in ctx.framing]]
    for layer in layers:
        if not layer.is_content or not layer.visible or layer.opacity <= 0:
            continue
        if ctx.vision is None:
            continue
        op = layer.operations[0]
        chain.append([layer.mask.model_dump() if layer.mask else None, cache_identity(op)])
        if isinstance(op, ops.Remove):
            x = _apply_removal(x, layer, op, [*chain, x.shape], ctx)
        elif isinstance(op, ops.GenerativeBase):
            x = _apply_generative(x, layer, op, chain, ctx)
    return x


def _chain_key(chain: list[object]) -> str:
    return hashlib.sha256(json.dumps(chain, default=str).encode()).hexdigest()[:24]


def _hole(layer: Layer, x: Array, grow: float, ctx: RenderContext) -> npt.NDArray[np.bool_] | None:
    """Where a content layer's mask selects, grown by `grow` (0..100), as a boolean map."""
    if layer.mask is None:
        return None
    region = ctx.mask(layer.mask, x)
    radius = max(1, round(grow / 100 * 0.02 * long_edge(x)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    hole = cv2.dilate((region > REMOVAL_THRESHOLD).astype(np.uint8), kernel).astype(bool)
    return hole if hole.any() else None


def _apply_removal(
    x: Array, layer: Layer, op: ops.Remove, chain: list[object], ctx: RenderContext
) -> Array:
    assert ctx.vision is not None
    hole = _hole(layer, x, op.grow, ctx)
    if hole is None:
        return x
    filled = ctx.vision.fill(np.clip(x, 0.0, 1.0), hole, _chain_key(chain))
    return _blend_hole(x, filled, hole, op.grow, layer.opacity)


def _blend_hole(
    x: Array, filled: Array, hole: npt.NDArray[np.bool_], grow: float, opacity: float
) -> Array:
    radius = max(1, round(grow / 100 * 0.02 * long_edge(x)))
    weight = blur(hole.astype(np.float32), max(1.0, radius / 2)) * (opacity / 100)
    return x + (filled - x) * weight[..., None]


def _apply_generative(
    x: Array, layer: Layer, op: ops.GenerativeBase, chain: list[object], ctx: RenderContext
) -> Array:
    assert ctx.vision is not None
    if isinstance(op, ops.Generate):
        hole = _hole(layer, x, op.grow, ctx)
        if hole is None:
            return x
        made = ctx.vision.generate(
            task_for(op), np.clip(x, 0.0, 1.0), hole, job_params(op), _chain_key(chain)
        )
        return _blend_hole(x, made, hole, op.grow, layer.opacity)
    if isinstance(op, ops.ReplaceBackground):
        return _replace_background(x, layer, op, chain, ctx)
    if isinstance(op, ops.Relight):
        made = ctx.vision.generate(
            task_for(op),
            np.clip(x, 0.0, 1.0),
            None,
            {**job_params(op), "direction": op.direction},
            _chain_key(chain),
        )
        return _blend_whole(x, transfer_light(x, made), layer, op.amount, ctx)
    if isinstance(op, ops.RestoreFaces):
        # Faces need more pixels than other generated results to keep their new detail.
        made = ctx.vision.generate(
            task_for(op),
            np.clip(x, 0.0, 1.0),
            None,
            job_params(op),
            _chain_key(chain),
            RESTORE_EDGE,
        )
        return _blend_whole(x, made, layer, op.amount, ctx)
    if isinstance(op, ops.Colorize):
        made = ctx.vision.generate(
            task_for(op), np.clip(x, 0.0, 1.0), None, job_params(op), _chain_key(chain)
        )
        return _blend_whole(x, keep_luminance(x, made), layer, op.amount, ctx)
    if isinstance(op, ops.Restyle):
        params = {**job_params(op), "strength": 0.25 + 0.7 * op.strength / 100, "restyle": True}
        made = ctx.vision.generate(
            task_for(op), np.clip(x, 0.0, 1.0), None, params, _chain_key(chain)
        )
        return _blend_whole(x, made, layer, 100, ctx)
    return x


RESTORE_EDGE = 2048
"""Working size for face restoration."""


def keep_luminance(x: Array, colored: Array) -> Array:
    """The photo's own brightness and detail with the color of `colored` (a colorized
    version of it, generated smaller)."""
    lab = cv2.cvtColor(np.clip(x, 0.0, 1.0).astype(np.float32), cv2.COLOR_RGB2LAB)
    other = cv2.cvtColor(np.clip(colored, 0.0, 1.0).astype(np.float32), cv2.COLOR_RGB2LAB)
    lab[..., 1:] = other[..., 1:]
    return cast(Array, np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), 0.0, 1.0))


def _blend_whole(x: Array, result: Array, layer: Layer, amount: float, ctx: RenderContext) -> Array:
    """Blend a whole-photo result in by amount (0..100), opacity, and the layer's mask."""
    weight: Array | float = amount / 100 * layer.opacity / 100
    if layer.mask is not None:
        weight = ctx.mask(layer.mask, x)[..., None] * weight
    return cast(Array, x + (result - x) * weight)


def transfer_light(x: Array, lit: Array) -> Array:
    """The photo's own fine detail under the broad light and color of `lit`, a relit
    version of it (generated smaller, so its own detail is soft or reinvented)."""
    sigma = max(1.0, long_edge(x) * 0.006)
    eps = 0.02
    base = blur(np.clip(x, 0.0, 1.0), sigma)
    light = blur(np.clip(lit, 0.0, 1.0), sigma)
    return cast(Array, np.clip(x * (light + eps) / (base + eps), 0.0, 1.5))


EVERYTHING_BUT_THE_SUBJECT = SemanticMask(target="subject", invert=True)


def _replace_background(
    x: Array, layer: Layer, op: ops.ReplaceBackground, chain: list[object], ctx: RenderContext
) -> Array:
    assert ctx.vision is not None
    region = np.clip(ctx.mask(layer.mask or EVERYTHING_BUT_THE_SUBJECT, x), 0.0, 1.0)
    hole = region > REMOVAL_THRESHOLD
    if not hole.any():
        return x
    params = {**job_params(op), "backdrop": True}
    made = ctx.vision.generate(task_for(op), np.clip(x, 0.0, 1.0), hole, params, _chain_key(chain))
    # The mask's soft edge (a matte, for the subject) decides the blend, so hair and fine
    # edges stay as they were.
    composite = x + (made - x) * region[..., None]
    composite = harmonize(composite, 1.0 - region, op.harmonize)
    return cast(Array, x + (composite - x) * (layer.opacity / 100))


def render_layer_mask(pixels: Array, state: EditState, layer_id: str, ctx: RenderContext) -> Array:
    """Where a layer applies (0..1 per pixel), for showing its mask. The id "cutout" shows
    what the cutout keeps. Raises KeyError."""
    if layer_id == CUTOUT_ID:
        if state.cutout is None:
            raise KeyError(layer_id)
        _, alpha = render_cutout(
            pixels,
            state.model_copy(update={"cutout": state.cutout.model_copy(update={"visible": True})}),
            ctx,
        )
        assert alpha is not None
        return alpha
    target = state.layer(layer_id)
    ctx = replace(ctx, framing=tuple(state.framing))
    out = apply_operations(pixels.astype(np.float32, copy=True), state.framing, ctx)
    mask = target.mask
    if mask is None and any(isinstance(op, ops.ReplaceBackground) for op in target.operations):
        mask = EVERYTHING_BUT_THE_SUBJECT
    if mask is None:
        return np.ones(out.shape[:2], np.float32)
    if not target.is_content:
        out = _apply_content(out, state.layers, ctx)
    for layer in state.layers:
        if layer.id == layer_id:
            break
        out = _apply_layer(out, layer, ctx)
    return ctx.mask(mask, out)


def apply_operations(x: Array, operations: Sequence[ops.OpBase], ctx: RenderContext) -> Array:
    for i, op in enumerate(operations):
        if isinstance(op, ops.Expand):
            # What it paints depends on the framing before it.
            x = _expand(x, op, ctx, operations[:i])
        else:
            x = _APPLY[type(op)](x, op, ctx)
    return x


def blend(base: Array, top: Array, mode: BlendMode) -> Array:
    """Combine a layer's adjusted pixels (`top`) with the pixels below it (`base`)."""
    if mode == "normal":
        return top
    if mode == "luminosity":
        return cast(Array, base + (luma(top) - luma(base))[..., None])
    if mode == "color":
        return cast(Array, top + (luma(base) - luma(top))[..., None])
    a, b = np.clip(base, 0.0, 1.0), np.clip(top, 0.0, 1.0)
    if mode == "multiply":
        return cast(Array, a * b)
    if mode == "screen":
        return cast(Array, 1.0 - (1.0 - a) * (1.0 - b))
    if mode == "overlay":
        return cast(Array, np.where(a < 0.5, 2.0 * a * b, 1.0 - 2.0 * (1.0 - a) * (1.0 - b)))
    # soft_light (the "pegtop" formula: smooth, no hard edge at mid-gray)
    return cast(Array, (1.0 - 2.0 * b) * a * a + 2.0 * b * a)


# Helpers


def luma(x: Array) -> Array:
    return cast(Array, (x @ LUMA).astype(np.float32))


def smoothstep(e0: float, e1: float, x: Array) -> Array:
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return cast(Array, (t * t * (3.0 - 2.0 * t)).astype(np.float32))


def _srgb_to_linear_exact(x: Array) -> Array:
    x = np.maximum(x, 0.0)
    return cast(Array, np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4))


def _linear_to_srgb_exact(x: Array) -> Array:
    x = np.maximum(x, 0.0)
    return cast(Array, np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055))


LUT_SIZE = 65536
_LUT_GRID = np.linspace(0.0, 1.0, LUT_SIZE, dtype=np.float32)
_S2L = _srgb_to_linear_exact(_LUT_GRID).astype(np.float32)
_L2S = _linear_to_srgb_exact(_LUT_GRID).astype(np.float32)


def apply_lut(x: Array, lut: Array, beyond: Callable[[Array], Array] | None = None) -> Array:
    """Map values in 0..1 through a lookup table; `beyond` handles values above 1 exactly
    (values below 0 map to the first entry)."""
    idx = (np.clip(x, 0.0, 1.0) * (lut.size - 1) + 0.5).astype(np.int32)
    out = lut[idx]
    if beyond is not None:
        over = x > 1.0
        if over.any():
            out[over] = beyond(x[over])
    return out


def srgb_to_linear(x: Array) -> Array:
    return apply_lut(x, _S2L, _srgb_to_linear_exact)


def linear_to_srgb(x: Array) -> Array:
    return apply_lut(x, _L2S, _linear_to_srgb_exact)


def blur(x: Array, sigma: float) -> Array:
    if sigma < 0.3:
        return x
    return cast(Array, cv2.GaussianBlur(x, (0, 0), sigmaX=sigma, borderType=cv2.BORDER_REFLECT))


def long_edge(x: Array) -> int:
    return int(max(x.shape[0], x.shape[1]))


def _apply_gain_linear(x: Array, stops: Array) -> Array:
    """Multiply light by 2**stops (per pixel) in linear space, preserving hue."""
    return linear_to_srgb(srgb_to_linear(x) * np.exp2(stops)[..., None])


# Light


def _exposure(x: Array, op: ops.Exposure, ctx: RenderContext) -> Array:
    return linear_to_srgb(srgb_to_linear(x) * (2.0**op.stops))


def _contrast(x: Array, op: ops.Contrast, ctx: RenderContext) -> Array:
    a = op.amount / 100
    c = np.clip(x, 0.0, 1.0)
    if a >= 0:
        s_curve = c * c * (3.0 - 2.0 * c)
        y = c + a * (s_curve - c)
    else:
        y = 0.5 + (c - 0.5) * (1.0 + 0.6 * a)
    return cast(Array, x + (y - c))


def _tonal_mask_luma(x: Array) -> Array:
    """Luminance softened toward its local average, so tonal masks follow regions not noise."""
    lum = np.clip(luma(x), 0.0, 1.0)
    return cast(Array, 0.5 * lum + 0.5 * blur(lum, long_edge(x) * 0.004))


def _highlights(x: Array, op: ops.Highlights, ctx: RenderContext) -> Array:
    mask = smoothstep(0.45, 1.0, _tonal_mask_luma(x))
    return _apply_gain_linear(x, mask * (op.amount / 100) * 1.2)


def _shadows(x: Array, op: ops.Shadows, ctx: RenderContext) -> Array:
    mask = 1.0 - smoothstep(0.0, 0.55, _tonal_mask_luma(x))
    return _apply_gain_linear(x, mask * (op.amount / 100) * 1.5)


def _whites(x: Array, op: ops.Whites, ctx: RenderContext) -> Array:
    weight = smoothstep(0.35, 1.0, np.clip(luma(x), 0.0, 1.0))[..., None]
    return cast(Array, x * (1.0 + (op.amount / 100) * 0.35 * weight))


def _blacks(x: Array, op: ops.Blacks, ctx: RenderContext) -> Array:
    weight = 1.0 - smoothstep(0.0, 0.4, np.clip(x, 0.0, 1.0))
    return cast(Array, x + (op.amount / 100) * 0.12 * weight)


# Color


def _white_balance(x: Array, op: ops.WhiteBalance, ctx: RenderContext) -> Array:
    t, tint = op.temperature / 100, op.tint / 100
    gains = np.array([1 + 0.25 * t, 1 - 0.2 * tint, 1 - 0.25 * t], dtype=np.float32)
    gains /= float(gains @ LUMA)
    return linear_to_srgb(srgb_to_linear(x) * gains)


def _saturation(x: Array, op: ops.Saturation, ctx: RenderContext) -> Array:
    lum = luma(x)[..., None]
    return cast(Array, lum + (x - lum) * (1.0 + op.amount / 100))


def _hsv(x: Array) -> Array:
    return cast(Array, cv2.cvtColor(np.clip(x, 0.0, 1.0), cv2.COLOR_RGB2HSV))


def _skin_weight(hue: Array, sat: Array) -> Array:
    """How much a pixel looks like a skin tone (orange hues, moderate saturation)."""
    hue_w = np.clip(1.0 - np.abs(hue - 25.0) / 25.0, 0.0, 1.0)
    return cast(Array, hue_w * smoothstep(0.1, 0.25, sat) * (1.0 - smoothstep(0.55, 0.8, sat)))


def _vibrance(x: Array, op: ops.Vibrance, ctx: RenderContext) -> Array:
    hsv = _hsv(x)
    sat = hsv[..., 1]
    weight = (1.0 - sat) * (1.0 - 0.6 * _skin_weight(hsv[..., 0], sat))
    factor = 1.0 + (op.amount / 100) * 1.2 * weight
    lum = luma(x)[..., None]
    return cast(Array, lum + (x - lum) * factor[..., None])


BAND_CENTERS = {
    "red": 0.0,
    "orange": 30.0,
    "yellow": 60.0,
    "green": 120.0,
    "aqua": 180.0,
    "blue": 225.0,
    "purple": 270.0,
    "magenta": 315.0,
}


def _hsl(x: Array, op: ops.HSL, ctx: RenderContext) -> Array:
    hsv = _hsv(x)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    dist = np.abs((hue - BAND_CENTERS[op.band] + 180.0) % 360.0 - 180.0)
    width = 30.0 if op.band in ("red", "orange", "yellow") else 45.0
    w = 0.5 + 0.5 * np.cos(np.pi * np.clip(dist / width, 0.0, 1.0))
    w = (w * smoothstep(0.03, 0.2, sat)).astype(np.float32)

    hsv[..., 0] = (hue + w * (op.hue / 100) * 30.0) % 360.0
    hsv[..., 1] = np.clip(sat * (1.0 + w * op.saturation / 100), 0.0, 1.0)
    hsv[..., 2] = val * np.exp2(w * (op.luminance / 100) * 0.8)
    rgb = cast(Array, cv2.cvtColor(hsv.astype(np.float32), cv2.COLOR_HSV2RGB))
    # Keep any out-of-range values from earlier steps that the HSV round trip clipped.
    return cast(Array, rgb + (x - np.clip(x, 0.0, 1.0)))


# Detail


def _sharpen(x: Array, op: ops.Sharpen, ctx: RenderContext) -> Array:
    lum = luma(x)
    detail = lum - blur(lum, max(0.35, op.radius * ctx.scale))
    return cast(Array, x + (op.amount / 100) * 1.5 * detail[..., None])


def _noise_reduction(x: Array, op: ops.NoiseReduction, ctx: RenderContext) -> Array:
    l_amt, c_amt = op.luminance / 100, op.color / 100
    ycc = cast(Array, cv2.cvtColor(np.clip(x, 0.0, 1.0), cv2.COLOR_RGB2YCrCb))
    if l_amt > 0:
        y = np.ascontiguousarray(ycc[..., 0])
        smooth = cv2.bilateralFilter(
            y, d=-1, sigmaColor=0.03 + 0.12 * l_amt, sigmaSpace=max(1.0, 3.0 * ctx.scale)
        )
        ycc[..., 0] = y + l_amt * (smooth - y)
    if c_amt > 0:
        sigma = max(1.0, 4.0 * c_amt * ctx.scale)
        for ch in (1, 2):
            channel = np.ascontiguousarray(ycc[..., ch])
            ycc[..., ch] = channel + min(1.0, 1.2 * c_amt) * (blur(channel, sigma) - channel)
    rgb = cast(Array, cv2.cvtColor(ycc, cv2.COLOR_YCrCb2RGB))
    return cast(Array, rgb + (x - np.clip(x, 0.0, 1.0)))


def _clarity(x: Array, op: ops.Clarity, ctx: RenderContext) -> Array:
    lum = np.clip(luma(x), 0.0, 1.0)
    detail = lum - blur(lum, long_edge(x) * 0.015)
    midtones = 1.0 - (2.0 * lum - 1.0) ** 2
    return x + ((op.amount / 100) * 1.2 * detail * midtones)[..., None]


def _dehaze(x: Array, op: ops.Dehaze, ctx: RenderContext) -> Array:
    a = op.amount / 100
    h, w = x.shape[:2]
    # Haze varies slowly across a scene, so estimate it on a small copy.
    small = np.clip(_shrink(x, 512), 0.0, 1.0)
    size = max(3, int(long_edge(small) * 0.015) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    dark = cv2.erode(small.min(axis=2), kernel)
    # Atmospheric light: average color of the haziest 0.1% of pixels.
    n = max(1, dark.size // 1000)
    idx = np.argpartition(dark.ravel(), -n)[-n:]
    atmos = np.maximum(small.reshape(-1, 3)[idx].mean(axis=0), 0.2).astype(np.float32)
    if a < 0:
        return cast(Array, x + (-a) * 0.5 * (float(atmos.mean()) - x))
    dark_norm = np.asarray(cv2.erode((small / atmos).min(axis=2), kernel), np.float32)
    trans_small = np.clip(1.0 - 0.95 * blur(dark_norm, size * 2.0), 0.15, 1.0)
    trans = cv2.resize(trans_small, (w, h), interpolation=cv2.INTER_LINEAR)[..., None]
    clear = (x - atmos) / trans + atmos
    return cast(Array, x + a * (clear - x))


def _shrink(x: Array, max_edge: int) -> Array:
    scale = max_edge / long_edge(x)
    if scale >= 1:
        return x
    size = (max(1, round(x.shape[1] * scale)), max(1, round(x.shape[0] * scale)))
    return cast(Array, cv2.resize(x, size, interpolation=cv2.INTER_AREA))


# Geometry

ASPECTS = {
    "1:1": 1.0,
    "4:5": 4 / 5,
    "5:4": 5 / 4,
    "3:4": 3 / 4,
    "4:3": 4 / 3,
    "2:3": 2 / 3,
    "3:2": 3 / 2,
    "9:16": 9 / 16,
    "16:9": 16 / 9,
}


def crop_box(op: ops.Crop, width: int, height: int, source_aspect: float) -> tuple[int, ...]:
    """The (left, top, right, bottom) pixel box a crop selects in a width x height frame."""
    left, top = op.left * width, op.top * height
    box_w, box_h = (op.right - op.left) * width, (op.bottom - op.top) * height
    ratio = source_aspect if op.aspect == "original" else ASPECTS.get(op.aspect)
    if ratio is not None:
        cx, cy = left + box_w / 2, top + box_h / 2
        if box_w / box_h > ratio:
            box_w = box_h * ratio
        else:
            box_h = box_w / ratio
        left, top = cx - box_w / 2, cy - box_h / 2
    w_px = min(width, max(1, round(box_w)))
    h_px = min(height, max(1, round(box_h)))
    x0 = min(width - w_px, max(0, round(left + (box_w - w_px) / 2)))
    y0 = min(height - h_px, max(0, round(top + (box_h - h_px) / 2)))
    x1, y1 = x0 + w_px, y0 + h_px
    return x0, y0, x1, y1


def _crop(x: Array, op: ops.Crop, ctx: RenderContext) -> Array:
    x0, y0, x1, y1 = crop_box(op, x.shape[1], x.shape[0], ctx.source_aspect)
    return cast(Array, np.ascontiguousarray(x[y0:y1, x0:x1]))


def _rotate(x: Array, op: ops.Rotate, ctx: RenderContext) -> Array:
    return cast(Array, np.ascontiguousarray(np.rot90(x, k=-op.degrees // 90)))


def _flip(x: Array, op: ops.Flip, ctx: RenderContext) -> Array:
    flipped = x[:, ::-1] if op.axis == "horizontal" else x[::-1]
    return cast(Array, np.ascontiguousarray(flipped))


def largest_rotated_rect(w: float, h: float, angle_rad: float) -> tuple[float, float]:
    """Size of the largest axis-aligned rectangle inside a w x h rectangle rotated by angle."""
    if w <= 0 or h <= 0:
        return 0.0, 0.0
    width_is_longer = w >= h
    side_long, side_short = (w, h) if width_is_longer else (h, w)
    sin_a, cos_a = abs(math.sin(angle_rad)), abs(math.cos(angle_rad))
    if side_short <= 2.0 * sin_a * cos_a * side_long or abs(sin_a - cos_a) < 1e-10:
        half = 0.5 * side_short
        wr, hr = (half / sin_a, half / cos_a) if width_is_longer else (half / cos_a, half / sin_a)
    else:
        cos_2a = cos_a * cos_a - sin_a * sin_a
        wr, hr = (w * cos_a - h * sin_a) / cos_2a, (h * cos_a - w * sin_a) / cos_2a
    return wr, hr


def _straighten(x: Array, op: ops.Straighten, ctx: RenderContext) -> Array:
    if abs(op.angle) < 1e-3:
        return x
    h, w = x.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), -op.angle, 1.0)
    rotated = cv2.warpAffine(
        x, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT
    )
    cw, ch = largest_rotated_rect(w, h, math.radians(op.angle))
    x0, y0 = math.ceil((w - cw) / 2), math.ceil((h - ch) / 2)
    x1, y1 = max(x0 + 1, w - x0), max(y0 + 1, h - y0)
    return cast(Array, np.ascontiguousarray(rotated[y0:y1, x0:x1]))


# How far the strongest perspective and lens corrections go.
PERSPECTIVE_MAX = 0.3
"""Fraction of the width (or height) the narrow side is stretched by at 100."""
DISTORTION_MAX = 0.25
"""Radial coefficient at 100, with the radius measured to the corners."""


def perspective_quad(op: ops.Perspective, w: int, h: int) -> Array:
    """The corners (top-left, top-right, bottom-right, bottom-left) of the region that gets
    stretched to fill the frame."""
    quad = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    kv, kh = PERSPECTIVE_MAX * op.vertical / 100, PERSPECTIVE_MAX * op.horizontal / 100
    top, bottom = (0, 1) if kv > 0 else (3, 2)
    quad[top, 0] += abs(kv) * w / 2
    quad[bottom, 0] -= abs(kv) * w / 2
    right, left = (1, 2) if kh > 0 else (0, 3)
    quad[right, 1] += abs(kh) * h / 2
    quad[left, 1] -= abs(kh) * h / 2
    return quad


def _perspective(x: Array, op: ops.Perspective, ctx: RenderContext) -> Array:
    if abs(op.vertical) < 1e-3 and abs(op.horizontal) < 1e-3:
        return x
    h, w = x.shape[:2]
    frame = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(perspective_quad(op, w, h), frame)
    out = cv2.warpPerspective(
        x, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return cast(Array, out)


def _lens_correction(x: Array, op: ops.LensCorrection, ctx: RenderContext) -> Array:
    if abs(op.distortion) < 1e-3:
        return x
    h, w = x.shape[:2]
    k = -DISTORTION_MAX * op.distortion / 100
    reach = math.hypot(w, h) / 2
    # Scale so the edge point that samples farthest out still lands inside the photo.
    nearest_edge = min(w, h) / 2 / reach
    scale = 1 / max(1 + k * nearest_edge**2, 1 + k)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    dx, dy = (xs - (w - 1) / 2) / reach, (ys - (h - 1) / 2) / reach
    factor = (1 + k * (dx * dx + dy * dy)) * scale
    map_x = (dx * factor * reach + (w - 1) / 2).astype(np.float32)
    map_y = (dy * factor * reach + (h - 1) / 2).astype(np.float32)
    out = cv2.remap(x, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return cast(Array, out)


def expand_box(op: ops.Expand, w: int, h: int) -> tuple[int, int, int, int]:
    """The expanded canvas size and where the photo sits in it: (width, height, x, y)."""
    if op.aspect != "free":
        ratio = ASPECTS[op.aspect]
        if w / h < ratio:
            new_w, new_h = max(w, round(h * ratio)), h
        else:
            new_w, new_h = w, max(h, round(w / ratio))
        return new_w, new_h, (new_w - w) // 2, (new_h - h) // 2
    left, right = round(op.left * w), round(op.right * w)
    top, bottom = round(op.top * h), round(op.bottom * h)
    return w + left + right, h + top + bottom, left, top


EXPAND_OVERLAP = 0.015
"""How far into the photo (fraction of the long edge) the model may repaint, so the seam
between the photo and the new area blends."""
EXPAND_PROMPT = "a natural continuation of the surrounding scene, photograph"


def extend_canvas(x: Array, new_w: int, new_h: int, x0: int, y0: int) -> Array:
    """The photo on a larger canvas, the new area holding a mirror of the photo that softens
    with distance: a starting point the generative model paints over."""
    h, w = x.shape[:2]
    pad = ((y0, new_h - h - y0), (x0, new_w - w - x0), (0, 0))
    mirrored = np.pad(np.clip(x, 0.0, 1.0), pad, mode="symmetric").astype(np.float32)
    edge = max(new_w, new_h)
    soft = blur(mirrored, edge * 0.02)
    inside = np.zeros((new_h, new_w), np.uint8)
    inside[y0 : y0 + h, x0 : x0 + w] = 1
    distance = np.asarray(cv2.distanceTransform(1 - inside, cv2.DIST_L2, 5), np.float32)
    fade = smoothstep(0.0, edge * 0.12, distance)[..., None]
    out = mirrored + (soft - mirrored) * fade
    out[y0 : y0 + h, x0 : x0 + w] = x
    return cast(Array, out)


def _expand(x: Array, op: ops.Expand, ctx: RenderContext, before: Sequence[ops.OpBase]) -> Array:
    h, w = x.shape[:2]
    new_w, new_h, x0, y0 = expand_box(op, w, h)
    if (new_w, new_h) == (w, h):
        return x
    canvas = extend_canvas(x, new_w, new_h, x0, y0)
    if ctx.vision is None:
        return canvas
    added = np.ones((new_h, new_w), np.uint8)
    added[y0 : y0 + h, x0 : x0 + w] = 0
    overlap = max(1, round(EXPAND_OVERLAP * max(new_w, new_h)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * overlap + 1, 2 * overlap + 1))
    hole = cv2.dilate(added, kernel).astype(bool)
    params = {**job_params(op), "prompt": op.prompt or EXPAND_PROMPT, "extend": True}
    key = _chain_key([[cache_identity(o) for o in before], cache_identity(op)])
    made = ctx.vision.generate("generate", canvas, hole, params, key)
    weight = np.maximum(blur(hole.astype(np.float32), overlap / 2), added.astype(np.float32))
    return cast(Array, canvas + (made - canvas) * weight[..., None])


# Finishing


def _vignette(x: Array, op: ops.Vignette, ctx: RenderContext) -> Array:
    h, w = x.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dist = np.sqrt(((xx + 0.5) / w - 0.5) ** 2 + ((yy + 0.5) / h - 0.5) ** 2) / math.sqrt(0.5)
    start = 0.15 + 0.6 * (op.midpoint / 100)
    mask = smoothstep(start, start + 0.65, dist)[..., None]
    a = op.amount / 100
    if a < 0:
        return cast(Array, x * (1.0 + 0.85 * a * mask))
    return cast(Array, x + (1.0 - x) * 0.8 * a * mask)


def _seed(op: ops.OpBase) -> int:
    return int.from_bytes(hashlib.sha256(op.id.encode()).digest()[:8], "little")


def hue_tint(hue: float) -> npt.NDArray[np.float32]:
    """The color offset that tints toward `hue` (degrees) without changing brightness."""
    hsv = np.array([[[hue % 360, 1.0, 1.0]]], np.float32)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0]
    return np.asarray(rgb - float(rgb @ LUMA), np.float32)


GRADE_STRENGTH = 0.22
"""How far a full-strength grade moves a tone toward its hue."""


def _color_grade(x: Array, op: ops.ColorGrade, ctx: RenderContext) -> Array:
    lum = np.clip(luma(x), 0.0, 1.0)
    pivot = 0.5 - op.balance / 100 * 0.25
    shadows = np.clip(1.0 - lum / pivot, 0.0, 1.0) ** 1.5
    highlights = np.clip((lum - pivot) / (1.0 - pivot), 0.0, 1.0) ** 1.5
    midtones = np.clip(1.0 - shadows - highlights, 0.0, 1.0) * 4 * lum * (1 - lum)
    out = x
    for weight, hue, amount in (
        (shadows, op.shadows_hue, op.shadows),
        (midtones, op.midtones_hue, op.midtones),
        (highlights, op.highlights_hue, op.highlights),
    ):
        if amount > 0:
            tint = hue_tint(hue) * (GRADE_STRENGTH * amount / 100)
            out = out + weight[..., None] * tint
    return out


def _grain(x: Array, op: ops.Grain, ctx: RenderContext) -> Array:
    h, w = x.shape[:2]
    grain_px = max(1.0, long_edge(x) * (0.0004 + 0.0016 * op.size / 100))
    gh, gw = max(2, round(h / grain_px)), max(2, round(w / grain_px))
    rng = np.random.default_rng(_seed(op))
    noise = rng.standard_normal((gh, gw)).astype(np.float32)
    noise = np.asarray(cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC), np.float32)
    lum = np.clip(luma(x), 0.0, 1.0)
    weight = 1.0 - 0.6 * (2.0 * lum - 1.0) ** 2
    return x + ((op.amount / 100) * 0.07 * noise * weight)[..., None]


# Retouching


def _smooth_skin(x: Array, op: ops.SmoothSkin, ctx: RenderContext) -> Array:
    """Edge-preserving smoothing (a guided filter, so edges like eyes and lips stay sharp)
    with the finest texture added back, so skin does not turn to plastic."""
    edge = long_edge(x)
    c = np.clip(x, 0.0, 1.0)
    radius = max(1, round(edge * 0.01))
    smooth = np.stack(
        [guided_filter(c[..., i], c[..., i], radius, eps=0.004) for i in range(3)], axis=2
    )
    fine = c - blur(c, max(0.5, edge * 0.0012))
    target = smooth + (op.texture / 100) * fine
    return cast(Array, x + (op.amount / 100) * (target - c))


def _heal_blemishes(x: Array, op: ops.HealBlemishes, ctx: RenderContext) -> Array:
    """Spots darker or redder than the skin around them, up to `size`, filled from their
    surroundings."""
    edge = long_edge(x)
    c = np.clip(x, 0.0, 1.0)
    max_radius = edge * (0.002 + 0.008 * op.size / 100)
    lum = luma(c)
    around = blur(lum, max_radius * 1.5)
    darker = around - blur(lum, max(0.5, max_radius * 0.25))
    red = c[..., 0] - 0.5 * (c[..., 1] + c[..., 2])
    redder = blur(red, max(0.5, max_radius * 0.25)) - blur(red, max_radius * 1.5)
    threshold = 0.06 - 0.045 * op.amount / 100
    spots = (darker > threshold) | (redder > threshold * 1.2)
    # Only small, round-ish spots: eyes, nostrils, and brows are bigger or longer. Thin
    # bridges are opened first so a blemish touching a crease still counts on its own.
    spots8 = spots.astype(np.uint8)
    if max_radius >= 2:
        cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        spots8 = cv2.morphologyEx(spots8, cv2.MORPH_OPEN, cross).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(spots8, connectivity=4)
    longest = stats[:, cv2.CC_STAT_WIDTH].clip(min=stats[:, cv2.CC_STAT_HEIGHT])
    keep = (longest <= 2 * max_radius + 1) & (
        stats[:, cv2.CC_STAT_AREA] <= math.pi * max_radius * max_radius
    )
    keep[0] = False
    small = keep[labels].astype(np.uint8)
    if count <= 1 or not small.any():
        return x
    grow = max(1, round(max_radius * 0.4))
    hole = cv2.dilate(small, np.ones((2 * grow + 1, 2 * grow + 1), np.uint8))
    img8 = (c * 255 + 0.5).astype(np.uint8)
    healed = cv2.inpaint(img8, hole, max(2.0, max_radius), cv2.INPAINT_TELEA).astype(np.float32)
    weight = blur(hole.astype(np.float32), max(0.5, grow / 2))[..., None]
    return cast(Array, x + (healed / 255 - c) * weight)


def monotone_curve(points: Sequence[Sequence[float]], samples: int = LUT_SIZE) -> Array:
    """Sample a monotone cubic (Fritsch-Carlson) through the points on a 0..1 grid."""
    pts: dict[float, float] = {}
    for px, py, *_ in sorted(points, key=lambda p: p[0]):
        pts[px] = py
    xs = np.array(list(pts.keys()), dtype=np.float64)
    ys = np.array(list(pts.values()), dtype=np.float64)
    grid = np.linspace(0.0, 1.0, samples)
    if len(xs) == 1:
        return np.full(samples, ys[0], dtype=np.float32)
    dx, dy = np.diff(xs), np.diff(ys)
    slopes = dy / dx
    tangents = np.empty_like(xs)
    tangents[0], tangents[-1] = slopes[0], slopes[-1]
    for i in range(1, len(xs) - 1):
        if slopes[i - 1] * slopes[i] <= 0:
            tangents[i] = 0.0
        else:
            tangents[i] = 2.0 / (1.0 / slopes[i - 1] + 1.0 / slopes[i])
    idx = np.clip(np.searchsorted(xs, grid) - 1, 0, len(xs) - 2)
    t = np.clip((grid - xs[idx]) / dx[idx], 0.0, 1.0)
    h00, h10 = 2 * t**3 - 3 * t**2 + 1, t**3 - 2 * t**2 + t
    h01, h11 = -2 * t**3 + 3 * t**2, t**3 - t**2
    out = h00 * ys[idx] + h10 * dx[idx] * tangents[idx] + h01 * ys[idx + 1]
    out += h11 * dx[idx] * tangents[idx + 1]
    out = np.where(grid < xs[0], ys[0], np.where(grid > xs[-1], ys[-1], out))
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _tone_curve(x: Array, op: ops.ToneCurve, ctx: RenderContext) -> Array:
    lut = monotone_curve(op.points)
    channels = {"rgb": [0, 1, 2], "red": [0], "green": [1], "blue": [2]}[op.channel]
    out = x.copy()
    for ch in channels:
        channel = x[..., ch]
        out[..., ch] = apply_lut(channel, lut) + (channel - np.clip(channel, 0.0, 1.0))
    return out


_APPLY: dict[type[ops.OpBase], Callable[[Array, Any, RenderContext], Array]] = {
    ops.Exposure: _exposure,
    ops.Contrast: _contrast,
    ops.Highlights: _highlights,
    ops.Shadows: _shadows,
    ops.Whites: _whites,
    ops.Blacks: _blacks,
    ops.WhiteBalance: _white_balance,
    ops.Vibrance: _vibrance,
    ops.Saturation: _saturation,
    ops.HSL: _hsl,
    ops.Sharpen: _sharpen,
    ops.NoiseReduction: _noise_reduction,
    ops.Clarity: _clarity,
    ops.Dehaze: _dehaze,
    ops.Crop: _crop,
    ops.Rotate: _rotate,
    ops.Straighten: _straighten,
    ops.Perspective: _perspective,
    ops.LensCorrection: _lens_correction,
    ops.Flip: _flip,
    ops.Vignette: _vignette,
    ops.Grain: _grain,
    ops.ToneCurve: _tone_curve,
    # Content operations need their layer's mask, so they render separately (see above).
    ops.Remove: lambda x, op, ctx: x,
    ops.Generate: lambda x, op, ctx: x,
    ops.ReplaceBackground: lambda x, op, ctx: x,
    ops.Relight: lambda x, op, ctx: x,
    ops.RestoreFaces: lambda x, op, ctx: x,
    ops.Colorize: lambda x, op, ctx: x,
    ops.Restyle: lambda x, op, ctx: x,
    ops.ColorGrade: _color_grade,
    # Expanding needs the framing before it, so `apply_operations` handles it.
    ops.Expand: lambda x, op, ctx: x,
    ops.SmoothSkin: _smooth_skin,
    ops.HealBlemishes: _heal_blemishes,
}


class RenderCache:
    """Remembers recent renders by document and edit state, so flipping between history
    steps or asking for the same preview twice does not re-render."""

    def __init__(self, size: int = 16) -> None:
        self._items: OrderedDict[tuple[str, str], Array] = OrderedDict()
        self._size = size
        self._lock = threading.Lock()

    def get_or_render(
        self,
        doc_id: str,
        pixels: Array,
        state: EditState,
        ctx: RenderContext,
    ) -> Array:
        key = (doc_id, state.fingerprint)
        with self._lock:
            hit = self._items.get(key)
            if hit is not None:
                self._items.move_to_end(key)
                return hit
        result = render_state(pixels, state, ctx)
        with self._lock:
            self._items[key] = result
            while len(self._items) > self._size:
                self._items.popitem(last=False)
        return result
