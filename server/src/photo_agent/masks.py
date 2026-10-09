"""Layer masks: limit a layer's effect to part of the photo.

A mask is data, like an operation. Positions are fractions of the framed photo (after crop
and rotation; 0 is the left or top edge, 1 the right or bottom), so a mask renders the same
on the preview proxy and at full resolution. Rendering a mask gives a per-pixel weight from
0 (layer has no effect) to 1 (full effect).

Semantic masks ("the sky", "the person on the left") are data too: what to select, not the
pixels. A model finds it in the framed original (see `photo_agent.vision.selection`), and the
result is cached, so the mask stays small, editable, and re-renders at any resolution.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, Any, Literal, cast

import cv2
import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, model_validator

from photo_agent.imaging import Array

Unit = Annotated[float, Field(ge=0, le=1)]
Point = Annotated[
    list[Annotated[float, Field(ge=-1, le=2)]],
    Field(min_length=2, max_length=2, description="[x, y] as fractions of the photo."),
]

LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

BRUSH_RENDER_LONG_EDGE = 1024
"""Brush masks are soft, so they are drawn at most this large and scaled up."""


class MaskBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    invert: bool = Field(False, description="Swap where the layer applies and where it does not.")


class BrushStroke(BaseModel):
    model_config = ConfigDict(extra="forbid")

    points: list[Point] = Field(min_length=1, max_length=4000)
    size: float = Field(
        ge=0.002, le=0.5, description="Brush radius as a fraction of the photo's long edge."
    )
    hardness: float = Field(50, ge=0, le=100, description="0 is a soft edge, 100 a hard one.")
    erase: bool = Field(False, description="Remove from the mask instead of adding to it.")


class BrushMask(MaskBase):
    """Painted by hand: the layer applies where the strokes are."""

    kind: Literal["brush"] = "brush"
    strokes: list[BrushStroke] = Field([], max_length=1000)


class LinearGradientMask(MaskBase):
    """Full effect on the `start` side, fading to none at `end`. For skies, use a start at
    the top and an end a little below the horizon."""

    kind: Literal["linear"] = "linear"
    start: Point
    end: Point


class RadialGradientMask(MaskBase):
    """Full effect inside an ellipse, fading out toward its edge. Invert it to affect
    everything outside instead (for example, to darken around a subject)."""

    kind: Literal["radial"] = "radial"
    center: Point
    radius_x: float = Field(gt=0, le=2, description="Half-width as a fraction of photo width.")
    radius_y: float = Field(gt=0, le=2, description="Half-height as a fraction of photo height.")
    feather: float = Field(50, ge=0, le=100, description="How gradually the edge fades.")


class LuminosityMask(MaskBase):
    """Applies to tones between `low` and `high` brightness (0 black, 1 white). For example
    0.65 to 1 targets highlights, 0 to 0.35 shadows."""

    kind: Literal["luminosity"] = "luminosity"
    low: Unit = 0
    high: Unit = 1
    feather: float = Field(0.1, ge=0, le=0.5, description="Softness of the range edges.")


SemanticTarget = Literal[
    "subject", "people", "sky", "object", "skin", "face", "eyes", "lips", "teeth", "hair"
]

TARGET_HELP = (
    "subject: the main subject, as a cutout would keep it; people: every person; sky; "
    "object: one thing you point at with `box` and/or `points` (a particular person, a dog, "
    "a car); and parts of faces for portraits: skin (face and visible skin), face, eyes, "
    "lips, teeth, hair."
)


class SelectPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x: Unit
    y: Unit
    include: bool = Field(True, description="False marks a spot that is not part of it.")


class SemanticMask(MaskBase):
    """Selects something by what it is, found by an AI model: the sky, the main subject,
    people, a particular object, or parts of a face. For `object`, give a `box` around it
    (and optionally `points` on it), in fractions of the photo as you see it."""

    kind: Literal["semantic"] = "semantic"
    target: SemanticTarget = Field(description=TARGET_HELP)
    box: list[Unit] | None = Field(
        None,
        min_length=4,
        max_length=4,
        description="For object: [left, top, right, bottom] around it, fractions of the photo.",
    )
    points: list[SelectPoint] = Field(
        [], max_length=50, description="For object: spots on it (or, with include false, not)."
    )
    description: str = Field(
        "", max_length=120, description='What is selected, in plain words: "the dog".'
    )

    @model_validator(mode="after")
    def _check_object(self) -> SemanticMask:
        if self.target == "object" and self.box is None and not self.points:
            raise ValueError("an object selection needs a box or points on the object")
        if self.box is not None and (self.box[2] <= self.box[0] or self.box[3] <= self.box[1]):
            raise ValueError("box must have right > left and bottom > top")
        return self


Mask = Annotated[
    BrushMask | LinearGradientMask | RadialGradientMask | LuminosityMask | SemanticMask,
    Field(discriminator="kind"),
]

SemanticResolver = Callable[[SemanticMask, tuple[int, int]], Array]
"""Finds a semantic mask's selection at a given (height, width) of the framed photo."""


class MaskUnavailableError(RuntimeError):
    """A semantic mask was rendered without a way to run the model that finds it."""


def render_mask(mask: Mask, x: Array, semantic: SemanticResolver | None = None) -> Array:
    """The mask's weight for every pixel of `x` (the pixels the layer applies to)."""
    h, w = x.shape[:2]
    if isinstance(mask, SemanticMask):
        if semantic is None:
            raise MaskUnavailableError("Semantic masks need the model worker.")
        alpha = np.asarray(semantic(mask, (h, w)), np.float32)
    elif isinstance(mask, BrushMask):
        alpha = _brush(mask, w, h)
    elif isinstance(mask, LinearGradientMask):
        alpha = _linear(mask, w, h)
    elif isinstance(mask, RadialGradientMask):
        alpha = _radial(mask, w, h)
    else:
        alpha = _luminosity(mask, x)
    return cast(Array, 1.0 - alpha) if mask.invert else alpha


def _smoothstep(e0: float, e1: float, v: npt.NDArray[Any]) -> Array:
    t = np.clip((v - e0) / (e1 - e0), 0.0, 1.0)
    return cast(Array, (t * t * (3.0 - 2.0 * t)).astype(np.float32))


def _grid(w: int, h: int) -> tuple[Array, Array]:
    """Pixel-center coordinates."""
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    return xs + 0.5, ys + 0.5


def _linear(mask: LinearGradientMask, w: int, h: int) -> Array:
    x0, y0 = mask.start[0] * w, mask.start[1] * h
    dx, dy = mask.end[0] * w - x0, mask.end[1] * h - y0
    length_sq = dx * dx + dy * dy
    if length_sq < 1e-6:
        return np.ones((h, w), np.float32)
    xs, ys = _grid(w, h)
    t = ((xs - x0) * dx + (ys - y0) * dy) / length_sq
    return cast(Array, 1.0 - _smoothstep(0.0, 1.0, t))


def _radial(mask: RadialGradientMask, w: int, h: int) -> Array:
    xs, ys = _grid(w, h)
    cx, cy = mask.center[0] * w, mask.center[1] * h
    d = np.sqrt(((xs - cx) / (mask.radius_x * w)) ** 2 + ((ys - cy) / (mask.radius_y * h)) ** 2)
    inner = 1.0 - mask.feather / 100
    if inner >= 1.0:
        return (d <= 1.0).astype(np.float32)
    return cast(Array, 1.0 - _smoothstep(inner, 1.0, d))


def _luminosity(mask: LuminosityMask, x: Array) -> Array:
    lum = cast(Array, np.clip(x @ LUMA, 0.0, 1.0).astype(np.float32))
    f = max(mask.feather, 1e-3)
    rise = _smoothstep(mask.low - f, mask.low, lum) if mask.low > 0 else 1.0
    fall = 1.0 - _smoothstep(mask.high, mask.high + f, lum) if mask.high < 1 else 1.0
    return cast(Array, np.asarray(rise * fall, np.float32) * np.ones_like(lum))


def _brush(mask: BrushMask, w: int, h: int) -> Array:
    scale = min(1.0, BRUSH_RENDER_LONG_EDGE / max(w, h))
    sw, sh = max(1, round(w * scale)), max(1, round(h * scale))
    long_edge = max(sw, sh)
    alpha = np.zeros((sh, sw), np.float32)
    for stroke in mask.strokes:
        radius = stroke.size * long_edge
        # Full strength out to `core`, fading to nothing at `radius`: draw a hard shape
        # halfway through the fade and blur it across the fade's width.
        core = radius * (0.4 + 0.6 * stroke.hardness / 100)
        drawn = (core + radius) / 2
        sigma = (radius - core) / 3
        dab = _draw_stroke(stroke.points, sw, sh, drawn)
        if sigma > 0.3:
            dab = np.asarray(cv2.GaussianBlur(dab, (0, 0), sigmaX=sigma), np.float32)
        if stroke.erase:
            alpha *= 1.0 - dab
        else:
            alpha = np.maximum(alpha, dab)
    if (sw, sh) != (w, h):
        alpha = np.asarray(cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR), np.float32)
    return cast(Array, alpha.astype(np.float32))


SUBPIXEL = 4
"""Bits of subpixel precision when drawing strokes."""


def _draw_stroke(points: list[list[float]], w: int, h: int, radius: float) -> Array:
    canvas = np.zeros((h, w), np.float32)
    k = 1 << SUBPIXEL
    pts = np.array([[round(p[0] * w * k), round(p[1] * h * k)] for p in points], np.int32)
    r = max(1, round(radius * k))
    if len(pts) > 1:
        thickness = max(1, round(radius * 2))
        cv2.polylines(canvas, [pts], False, 1.0, thickness, cv2.LINE_AA, SUBPIXEL)
    for x, y in (pts[0], pts[-1]):
        cv2.circle(canvas, (int(x), int(y)), r, 1.0, -1, cv2.LINE_AA, SUBPIXEL)
    return canvas


def describe_mask(mask: Mask) -> str:
    """A short plain description for the agent and for summaries."""
    if isinstance(mask, BrushMask):
        text = f"brush mask ({len(mask.strokes)} strokes)"
    elif isinstance(mask, LinearGradientMask):
        text = f"linear gradient mask from {_pt(mask.start)} to {_pt(mask.end)}"
    elif isinstance(mask, RadialGradientMask):
        text = (
            f"radial gradient mask at {_pt(mask.center)}, radius "
            f"{mask.radius_x:.2f}x{mask.radius_y:.2f}, feather {mask.feather:g}"
        )
    elif isinstance(mask, SemanticMask):
        what = mask.description or mask.target
        text = f"semantic mask selecting {what}"
        if mask.target == "object" and mask.box is not None:
            text += f" in box {_pt(mask.box[:2])}-{_pt(mask.box[2:])}"
    else:
        text = f"luminosity mask {mask.low:.2f}-{mask.high:.2f}"
    return f"inverted {text}" if mask.invert else text


def _pt(p: list[float]) -> str:
    return f"({p[0]:.2f}, {p[1]:.2f})"
