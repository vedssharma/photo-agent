"""How good a shot is, by simple measurements: sharpness, exposure, open eyes, and a rough
sense of appeal (color and contrast). Used to pick the best of a burst or a shoot.

Each measure is from 0 (bad) to 1 (good). They are meant for comparing shots of the same
moment with each other, where they are reliable (the sharper of two frames, the one where
nobody blinked), more than for judging photos of different things against each other.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from pydantic import BaseModel

from photo_agent import diagnostics
from photo_agent.imaging import Array, resize_long_edge
from photo_agent.vision import classical

WORK_EDGE = 1024


class Quality(BaseModel):
    sharpness: float
    """How crisp the most detailed part of the photo is (the subject, usually)."""
    exposure: float
    """1 when nothing is blown out or crushed and brightness is sensible."""
    appeal: float
    """A rough sense of visual appeal from color and contrast; a model replaces it when
    one is available."""
    faces: int
    eyes_closed: int
    """Faces whose eyes look closed."""


def sharpness(gray: Array) -> float:
    """Detail in the sharpest parts: the 90th percentile of local Laplacian variance, so a
    sharp subject on a soft background still counts as sharp."""
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    h, w = gray.shape
    tiles = [
        float(lap[y : y + h // 8, x : x + w // 8].var())
        for y in range(0, h - h // 8 + 1, h // 8)
        for x in range(0, w - w // 8 + 1, w // 8)
    ]
    detail = max(float(np.percentile(tiles, 90)), 1e-6)
    # On a log scale: 1e-4 is mush, 0.3 is crisp at this size.
    return float(np.clip((np.log10(detail) + 4) / 3.5, 0, 1))


def exposure(pixels: Array) -> float:
    m = diagnostics.measure(pixels)
    penalty = (
        3 * max(0.0, m.clipped_highlights - 0.01)
        + 3 * max(0.0, m.crushed_shadows - 0.02)
        + 1.5 * max(0.0, abs(m.mean_brightness - 0.45) - 0.15)
    )
    return float(np.clip(1 - penalty, 0, 1))


def appeal(pixels: Array, gray: Array) -> float:
    """Colorfulness (Hasler and Süsstrunk) and contrast, each saturating."""
    r, g, b = pixels[..., 0], pixels[..., 1], pixels[..., 2]
    rg, yb = r - g, 0.5 * (r + g) - b
    colorful = float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean()))
    contrast = float(gray.std())
    return float(0.5 * min(1.0, colorful / 0.25) + 0.5 * min(1.0, contrast / 0.22))


def eyes_open(gray: Array, box: tuple[float, float, float, float]) -> float:
    """How open the eyes of a face look, from 0 to 1. Open eyes show dark irises and pupils
    against the white of the eye and the skin; closed eyes show lids the color of skin
    with a thin dark line. Looks for dark, roughly round spots in the eye band."""
    x0, y0, x1, y1 = (int(v) for v in box)
    face = gray[y0:y1, x0:x1]
    fh, fw = face.shape[:2]
    if fh < 24 or fw < 24 or sharpness(face) < 0.7:
        return 1.0  # too small or soft to tell; assume open
    band = face[int(fh * 0.2) : int(fh * 0.55), int(fw * 0.1) : int(fw * 0.9)]
    skin = float(np.median(face))
    dark = (band < skin * 0.45).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    eyes = 0
    for i in range(1, count):
        _, _, bw, bh, area = (int(v) for v in stats[i])
        # An open eye's dark part is about as tall as it is wide; a closed lid is a line.
        if area >= max(4, fw * fh * 0.002) and bh >= 0.45 * bw:
            eyes += 1
    return min(1.0, eyes / 2)


def measure(image: Array, **_: Any) -> Quality:
    pixels = np.ascontiguousarray(resize_long_edge(np.clip(image, 0, 1), WORK_EDGE))
    gray = np.asarray(cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY), np.float32)
    boxes = classical.find_faces(pixels)
    closed = sum(1 for box in boxes if eyes_open(gray, box) < 0.5)
    return Quality(
        sharpness=sharpness(gray),
        exposure=exposure(pixels),
        appeal=appeal(pixels, gray),
        faces=len(boxes),
        eyes_closed=closed,
    )
