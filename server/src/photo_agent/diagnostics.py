"""Measurements of a render that the agent uses to catch its own overshoots.

These are deliberately simple statistics. They are not a judgment of taste; they flag the
failures that are easy to cause with sliders and easy to miss in a small preview: blown
highlights, crushed shadows, and oversaturated color.
"""

from __future__ import annotations

import cv2
import numpy as np
from pydantic import BaseModel

from photo_agent.imaging import Array
from photo_agent.render import luma


class Measurements(BaseModel):
    clipped_highlights: float
    """Fraction of pixels with a channel at full brightness."""
    crushed_shadows: float
    """Fraction of pixels that are pure black."""
    oversaturated: float
    """Fraction of reasonably bright pixels with nearly maximal saturation."""
    mean_brightness: float


def measure(pixels: Array) -> Measurements:
    x = np.clip(pixels, 0.0, 1.0)
    lum = luma(x)
    hsv = cv2.cvtColor(x, cv2.COLOR_RGB2HSV)
    sat, val = hsv[..., 1], hsv[..., 2]
    return Measurements(
        clipped_highlights=float((x.max(axis=2) >= 0.995).mean()),
        crushed_shadows=float((lum <= 0.004).mean()),
        oversaturated=float(((sat >= 0.92) & (val >= 0.25)).mean()),
        mean_brightness=float(lum.mean()),
    )


def warnings(before: Measurements, after: Measurements) -> list[str]:
    """Problems the edits introduced, compared with the unedited photo."""
    found = []
    if after.clipped_highlights > max(0.02, before.clipped_highlights * 1.5 + 0.01):
        found.append(
            f"{after.clipped_highlights:.1%} of the photo is now blown-out white "
            f"(was {before.clipped_highlights:.1%})"
        )
    if after.crushed_shadows > max(0.02, before.crushed_shadows * 1.5 + 0.01):
        found.append(
            f"{after.crushed_shadows:.1%} of the photo is now pure black "
            f"(was {before.crushed_shadows:.1%})"
        )
    if after.oversaturated > max(0.03, before.oversaturated + 0.02):
        found.append(
            f"{after.oversaturated:.1%} of the photo is now at maximum saturation "
            f"(was {before.oversaturated:.1%}), which tends to look garish"
        )
    return found


def report(before: Measurements, after: Measurements) -> str:
    lines = [
        "Measurements after your edits (unedited photo in parentheses):",
        f"- blown-out highlights: {after.clipped_highlights:.1%} ({before.clipped_highlights:.1%})",
        f"- pure black shadows: {after.crushed_shadows:.1%} ({before.crushed_shadows:.1%})",
        f"- maximally saturated: {after.oversaturated:.1%} ({before.oversaturated:.1%})",
        f"- average brightness: {after.mean_brightness:.2f} ({before.mean_brightness:.2f})",
    ]
    problems = warnings(before, after)
    if problems:
        lines.append("Possible overshoots: " + "; ".join(problems) + ".")
    else:
        lines.append("No overshoots detected.")
    return "\n".join(lines)
