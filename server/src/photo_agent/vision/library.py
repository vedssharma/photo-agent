"""Classical stand-ins for the library's vision model: tags and a similarity embedding.

Without model weights the library still tags what simple measurements can tell (sky, faces,
snow, sunsets, night, greenery, water, black and white) and describes each photo with a
small feature vector (colors, layout, and texture) that is good enough to find near
duplicates and similar shots, though not to search by words.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from photo_agent.imaging import Array, resize_long_edge
from photo_agent.vision import classical

WORK_EDGE = 256


def _features(image: Array) -> tuple[Array, Array]:
    work = np.ascontiguousarray(resize_long_edge(np.clip(image, 0, 1), WORK_EDGE))
    hsv = np.asarray(cv2.cvtColor(work, cv2.COLOR_RGB2HSV), np.float32)
    return work, hsv


def tag(image: Array, **_: Any) -> dict[str, float]:
    """Tags simple measurements can vouch for, with a confidence from 0 to 1."""
    work, hsv = _features(image)
    h = work.shape[0]
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    found: dict[str, float] = {}

    sky = float(classical.sky(work).mean())
    if sky > 0.12:
        found["sky"] = min(1.0, sky * 2.5)

    faces = classical.find_faces(work)
    if faces:
        found["people"] = 0.8
        area = max((x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in faces) / work[..., 0].size
        if len(faces) >= 3:
            found["group"] = 0.7
        elif area > 0.03:
            found["portrait"] = min(1.0, 0.5 + area * 5)

    mean_sat, mean_val = float(sat.mean()), float(val.mean())
    if mean_sat < 0.04:
        found["black and white"] = 1.0 - mean_sat * 10
    elif mean_sat > 0.45 and mean_val > 0.3:  # dark noise is saturated but not colorful
        found["colorful"] = min(1.0, mean_sat)
    if mean_val < 0.2:
        found["night"] = min(1.0, (0.2 - mean_val) * 8 + 0.3)

    top = slice(0, h // 2)
    warm = ((hue < 45) | (hue > 330)) & (sat > 0.35) & (val > 0.35)
    if float(warm[top].mean()) > 0.3 and mean_val < 0.6:
        found["sunset"] = min(1.0, float(warm[top].mean()) * 1.5)

    snowy = (val > 0.8) & (sat < 0.12)
    if float(snowy[h // 3 :].mean()) > 0.4:
        found["snow"] = min(1.0, float(snowy.mean()) * 1.5)

    green = (hue > 70) & (hue < 160) & (sat > 0.2) & (val > 0.15)
    if float(green.mean()) > 0.25:
        found["nature"] = min(1.0, float(green.mean()) * 2)

    bottom = slice(h // 2, h)
    water = (hue > 170) & (hue < 230) & (sat > 0.2) & (val > 0.2)
    sand = (hue > 25) & (hue < 55) & (sat > 0.15) & (sat < 0.55) & (val > 0.5)
    if float(water[bottom].mean()) > 0.2:
        found["ocean"] = min(1.0, float(water[bottom].mean()) * 2)
        if float(sand[bottom].mean()) > 0.1 or "sky" in found:
            found["beach"] = 0.5
    return found


def embed(image: Array, **_: Any) -> list[float]:
    """A unit-length vector describing the photo's colors, layout, and texture."""
    work, hsv = _features(image)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 3, 3], [0, 360, 0, 1, 0, 1]).flatten()
    hist = np.sqrt(hist / max(float(hist.sum()), 1.0))  # Hellinger: tames dominant colors
    small = cv2.resize(work, (6, 6), interpolation=cv2.INTER_AREA).reshape(-1)
    small = (small - small.mean()) * 1.5
    gray = cv2.cvtColor(work, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    edges = np.histogram(angle % 180, bins=8, range=(0, 180), weights=magnitude)[0]
    edges = np.sqrt(edges / max(float(edges.sum()), 1e-6)) * 0.7
    vector = np.concatenate([hist, small, edges]).astype(np.float32)
    vector /= max(float(np.linalg.norm(vector)), 1e-6)
    return [float(v) for v in vector]


FACE_SIZE = 24


def faces(image: Array, **_: Any) -> list[dict[str, Any]]:
    """Faces (boxes in fractions of the photo) with a rough description of each: the
    face's evened-out grayscale pattern and its colors. Enough to group the same person
    across a shoot, not to recognize them across years."""
    h, w = image.shape[:2]
    found = []
    for x0, y0, x1, y1 in classical.find_faces(image):
        crop = np.clip(image[int(y0) : int(y1), int(x0) : int(x1)], 0, 1)
        if crop.size == 0 or min(crop.shape[:2]) < 12:
            continue
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        small = cv2.resize(gray, (FACE_SIZE, FACE_SIZE), interpolation=cv2.INTER_AREA)
        small = cv2.equalizeHist((small * 255).astype(np.uint8)).astype(np.float32) / 255
        pattern = (small - small.mean()).reshape(-1)
        pattern /= max(float(np.linalg.norm(pattern)), 1e-6)
        hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [8, 4], [0, 360, 0, 1]).flatten()
        hist = np.sqrt(hist / max(float(hist.sum()), 1.0)) * 0.5
        vector = np.concatenate([pattern, hist]).astype(np.float32)
        vector /= max(float(np.linalg.norm(vector)), 1e-6)
        found.append(
            {
                "box": [x0 / w, y0 / h, x1 / w, y1 / h],
                "embedding": [float(v) for v in vector],
                "confidence": 0.5,
            }
        )
    return found


def tag_photo(image: Array, **params: Any) -> dict[str, Any]:
    """Tags and an embedding in one go, as the model backend returns them."""
    return {"tags": tag(image), "embedding": embed(image), "text_search": False}
