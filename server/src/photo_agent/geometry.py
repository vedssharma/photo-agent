"""Finding how much a photo is tilted and how much its verticals converge, from the straight
lines in it, so the framing can be levelled and squared up automatically."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field

from photo_agent import operations as ops
from photo_agent.imaging import Array
from photo_agent.render import PERSPECTIVE_MAX

WORK_EDGE = 1024
MIN_LENGTH = 0.04
"""Shortest line segment considered, as a fraction of the long edge."""
MIN_LEVEL_LENGTH = 0.08
"""Shortest segment trusted for levelling: horizons and edges are long, whiskers and
hair are not."""
MAX_TILT = 20.0
"""Lines further than this many degrees from level or plumb are not horizons or walls."""
MIN_EVIDENCE = 0.5
"""Total length of agreeing lines, in long edges, needed to trust a tilt."""
MIN_VERTICALS = 4


def segments(image: Array) -> npt.NDArray[np.float64]:
    """Straight line segments as rows of (x0, y0, x1, y1), in pixels of a copy scaled so
    its long edge is 1, longest first."""
    h, w = image.shape[:2]
    scale = min(1.0, WORK_EDGE / max(h, w))
    gray = cv2.cvtColor(np.clip(image, 0, 1).astype(np.float32), cv2.COLOR_RGB2GRAY)
    if scale < 1:
        size = (round(w * scale), round(h * scale))
        gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    lines = cv2.createLineSegmentDetector().detect((gray * 255).astype(np.uint8))[0]
    if lines is None:
        return np.zeros((0, 4))
    found = lines.reshape(-1, 4).astype(np.float64) / max(gray.shape)
    length = np.hypot(found[:, 2] - found[:, 0], found[:, 3] - found[:, 1])
    keep = length >= MIN_LENGTH
    ordered: npt.NDArray[np.float64] = found[keep][np.argsort(-length[keep])]
    return ordered


def tilt(image: Array) -> float | None:
    """The Straighten angle that levels the photo, from its near-level and near-plumb lines,
    or None when there are too few to tell."""
    lines = segments(image)
    dx, dy = lines[:, 2] - lines[:, 0], lines[:, 3] - lines[:, 1]
    length = np.hypot(dx, dy)
    keep = length >= MIN_LEVEL_LENGTH
    if not keep.any():
        return None
    dx, dy, length = dx[keep], dy[keep], length[keep]
    angle = np.degrees(np.arctan2(dy, dx))
    angle = (angle + 90) % 180 - 90  # direction does not matter: -90..90
    level = np.abs(angle) <= MAX_TILT
    # A plumb line's lean, measured the same way as a level line's slope.
    plumb = np.abs(angle) >= 90 - MAX_TILT
    lean = np.where(angle > 0, angle - 90, angle + 90)
    devs = np.concatenate([angle[level], lean[plumb]])
    weights = np.concatenate([length[level], length[plumb]])
    if weights.sum() < MIN_EVIDENCE:
        return None
    # The peak of a length-weighted histogram: the tilt most lines agree on.
    step = 0.1
    bins = np.arange(-MAX_TILT, MAX_TILT + step, step)
    hist, _ = np.histogram(devs, bins=bins, weights=weights)
    hist = cv2.GaussianBlur(hist.astype(np.float32).reshape(1, -1), (0, 0), 3).ravel()
    peak = int(np.argmax(hist))
    near = np.abs(devs - (bins[peak] + step / 2)) <= 1.0
    if weights[near].sum() < MIN_EVIDENCE * 0.6:
        return None
    found = float(np.average(devs[near], weights=weights[near]))
    return -round(found, 2)


def keystone(image: Array) -> float | None:
    """The Perspective vertical amount that makes converging verticals parallel, or None
    when there are too few verticals away from the middle to tell."""
    h, w = image.shape[:2]
    lines = segments(image)
    if len(lines) == 0:
        return None
    long = max(h, w)
    x0, y0, x1, y1 = (lines[:, i] * long for i in range(4))
    down = y1 >= y0
    x0, y0, x1, y1 = (
        np.where(down, x0, x1),
        np.where(down, y0, y1),
        np.where(down, x1, x0),
        np.where(down, y1, y0),
    )
    dy = y1 - y0
    plumb = dy > 0
    slope = np.divide(x1 - x0, dy, out=np.zeros_like(dy), where=plumb)
    plumb &= np.abs(slope) <= math.tan(math.radians(MAX_TILT))
    offset = (x0 + x1) / 2 - w / 2
    # Lines near the middle say little about how fast the verticals converge.
    plumb &= np.abs(offset) >= 0.15 * w
    if plumb.sum() < MIN_VERTICALS:
        return None
    # Each line's slope is about offset * k / (h * (1 - |k| / 2)) for the warp Perspective
    # undoes; solve for k and take the length-weighted median.
    c = slope[plumb] / offset[plumb] * h
    k = c / (1 + np.abs(c) / 2)
    weights = np.hypot(x1 - x0, dy)[plumb] / long
    order = np.argsort(k)
    cumulative = np.cumsum(weights[order])
    median = float(k[order][np.searchsorted(cumulative, cumulative[-1] / 2)])
    # Trust it only when walls on both sides agree, as they do in buildings and rooms, not
    # when trees or limbs happen to lean.
    agree = np.abs(k - median) <= 0.04
    left = agree & (offset[plumb] < 0)
    right = agree & (offset[plumb] > 0)
    if (
        weights[agree].sum() < max(MIN_EVIDENCE, 0.6 * weights.sum())
        or min(weights[left].sum(), weights[right].sum()) < 0.25 * weights[agree].sum()
    ):
        return None
    return round(float(np.clip(median / PERSPECTIVE_MAX * 100, -100, 100)), 1)


@dataclass
class Leveled:
    framing: list[ops.OpBase]
    angle: float | None
    """Straighten angle found, or None when the photo gave no clear lines."""
    vertical: float | None
    """Perspective vertical amount found, or None when there were too few verticals."""

    def describe(self) -> str:
        """What was changed, or an empty string when the photo was already straight."""
        parts = []
        for op in self.framing:
            if isinstance(op, ops.Straighten) and op.angle == self.angle:
                parts.append(f"straighten {op.angle:+g}°")
            elif isinstance(op, ops.Perspective) and op.vertical == self.vertical:
                parts.append(f"perspective vertical {op.vertical:+g}")
        return ", ".join(parts)


class AutoStraighten(BaseModel):
    """What to fix automatically."""

    model_config = ConfigDict(extra="forbid")

    level: bool = Field(True, description="Level a tilted horizon or tilted verticals.")
    perspective: bool = Field(
        True, description="Make converging verticals (buildings, walls) parallel."
    )


Framed = Callable[[Sequence[ops.OpBase]], Array]
"""Renders the photo with the given framing."""

MIN_ANGLE = 0.1
MIN_VERTICAL = 3.0


def auto_level(
    framing: Sequence[ops.OpBase], framed: Framed, options: AutoStraighten | None = None
) -> Leveled:
    """New framing with a Straighten (and a Perspective) found from the photo, in place of
    any earlier ones, placed after rotations and flips but before the crop."""
    options = options or AutoStraighten()
    level, perspective = options.level, options.perspective
    replaced: tuple[type[ops.OpBase], ...] = ()
    if level:
        replaced += (ops.Straighten,)
    if perspective:
        replaced += (ops.Perspective,)
    kept = [op for op in framing if not isinstance(op, replaced)]
    at = next((i for i, op in enumerate(kept) if isinstance(op, ops.Crop)), len(kept))
    before = kept[:at]
    added: list[ops.OpBase] = []
    angle = vertical = None
    if level:
        angle = tilt(framed(before))
        if angle is not None and abs(angle) >= MIN_ANGLE:
            added.append(ops.Straighten(angle=angle))
    if perspective:
        vertical = keystone(framed([*before, *added]))
        if vertical is not None and abs(vertical) >= MIN_VERTICAL:
            added.append(ops.Perspective(vertical=vertical))
    return Leveled([*before, *added, *kept[at:]], angle, vertical)
