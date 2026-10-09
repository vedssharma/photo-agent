"""Compositing: make a subject sit naturally in a new background.

A cut-out subject pasted onto a new scene gives itself away: its colors and brightness come
from another light, and its edge is too clean. `harmonize` nudges the subject's overall
tone and color toward the new background and lets the background's light spill a little
over the subject's edge ("light wrap"), as it would in a real photo. It is deterministic
and cheap, so it runs at render time at any resolution.
"""

from __future__ import annotations

from typing import cast

import cv2
import numpy as np

from photo_agent.imaging import Array

TONE_SHARE = 0.45
"""How far the subject's average brightness moves toward the background's, at full amount."""
COLOR_SHARE = 0.6
"""How far the subject's average color cast moves toward the background's."""
WRAP = 0.5
"""Strength of the light wrap at full amount."""


def harmonize(image: Array, subject: Array, amount: float) -> Array:
    """`image` (RGB 0..1) with the `subject` (weights 0..1) matched to the rest.
    `amount` is 0..100; 0 returns the image unchanged."""
    a = amount / 100
    if a <= 0:
        return image
    x = np.clip(image, 0.0, 1.0).astype(np.float32)
    s = np.clip(subject, 0.0, 1.0).astype(np.float32)
    background = 1.0 - s
    if s.sum() < 1 or background.sum() < 1:
        return image
    lab = cv2.cvtColor(x, cv2.COLOR_RGB2LAB)
    fg_mean = (lab * s[..., None]).sum(axis=(0, 1)) / s.sum()
    bg_mean = (lab * background[..., None]).sum(axis=(0, 1)) / background.sum()
    shift = (bg_mean - fg_mean) * np.array([TONE_SHARE, COLOR_SHARE, COLOR_SHARE], np.float32)
    matched = cv2.cvtColor((lab + shift * a).astype(np.float32), cv2.COLOR_LAB2RGB)
    out = x + (np.clip(matched, 0.0, 1.0) - x) * s[..., None]
    return _light_wrap(out, s, a)


def _light_wrap(x: Array, subject: Array, a: float) -> Array:
    """Spill the background's light over the inside of the subject's edge."""
    h, w = subject.shape
    sigma = max(1.0, max(h, w) * 0.008)
    background = 1.0 - subject
    spread = cv2.GaussianBlur(background, (0, 0), sigma)
    light = cv2.GaussianBlur(x * background[..., None], (0, 0), sigma)
    light = light / np.maximum(spread, 1e-3)[..., None]
    # Strongest just inside the edge, nothing deep in the subject or out in the background.
    edge = np.clip(spread * subject * 2.0, 0.0, 1.0)[..., None] * (WRAP * a)
    screen = 1.0 - (1.0 - x) * (1.0 - np.clip(light, 0.0, 1.0))
    return cast(Array, x + (screen - x) * edge)
