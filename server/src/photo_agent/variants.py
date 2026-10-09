"""Variants: several takes on one generative edit, to pick from ("show me 3 options").

Each take is the same operation with another seed. The seeds on offer are recorded on the
operation (`options`), so the person can come back and compare them; picking one sets the
operation's seed. Generated results are cached by seed, so switching between takes that
were already shown is instant.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from photo_agent.generative import new_seed
from photo_agent.imaging import Array
from photo_agent.layers import EditState
from photo_agent.operations import MAX_OPTIONS, GenerativeBase

DEFAULT_COUNT = 3


class NotGenerativeError(KeyError):
    pass


def generative_op(state: EditState, op_id: str) -> GenerativeBase:
    """The generative operation with this id, wherever it is."""
    for op in state.all_operations():
        if op.id == op_id:
            if isinstance(op, GenerativeBase):
                return op
            break
    raise NotGenerativeError(op_id)


def offer(state: EditState, op_id: str, count: int = DEFAULT_COUNT) -> EditState:
    """The state with `count` takes on offer for the operation: the current one first, then
    new seeds."""
    if not 2 <= count <= MAX_OPTIONS:
        raise ValueError(f"count must be between 2 and {MAX_OPTIONS}")
    out = state.model_copy(deep=True)
    op = generative_op(out, op_id)
    current = op.seed if op.seed is not None else new_seed()
    op.seed = current
    seeds = [current]
    while len(seeds) < count:
        seed = new_seed()
        if seed not in seeds:
            seeds.append(seed)
    op.options = seeds
    return out


def with_seed(state: EditState, op_id: str, seed: int) -> EditState:
    """The state with the operation's seed set to one of its takes."""
    out = state.model_copy(deep=True)
    generative_op(out, op_id).seed = seed
    return out


def contact_sheet(images: Sequence[Array], gap: int = 8) -> Array:
    """The takes side by side, numbered from 1, at a common height."""
    height = min(image.shape[0] for image in images)
    tiles = []
    for i, image in enumerate(images, start=1):
        width = max(1, round(image.shape[1] * height / image.shape[0]))
        small = cv2.resize(np.clip(image, 0, 1), (width, height), interpolation=cv2.INTER_AREA)
        tile = np.ascontiguousarray((small * 255 + 0.5).astype(np.uint8))
        size = max(0.6, height / 300)
        org = (round(10 * size), round(34 * size))
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(tile, str(i), org, font, size * 1.2, (0, 0, 0), 6, cv2.LINE_AA)
        cv2.putText(tile, str(i), org, font, size * 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(tile.astype(np.float32) / 255)
        tiles.append(np.ones((height, gap, 3), np.float32))
    return np.concatenate(tiles[:-1], axis=1).astype(np.float32)
