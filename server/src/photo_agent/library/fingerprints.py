"""A fingerprint of each photo's look, for finding duplicates (see `duplicates.py`).

The fingerprint is a 64-bit difference hash: the photo shrunk to 9 by 8 grays, one bit per
pair of neighbors saying which is brighter. It stays the same, or nearly, when a photo is
resized, re-saved, or lightly edited.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from photo_agent import imaging

if TYPE_CHECKING:
    from photo_agent.library.core import Library


def fingerprint(image: imaging.Array) -> str:
    """A difference hash of the photo, as 16 hex digits."""
    gray = cv2.cvtColor(np.ascontiguousarray(np.clip(image, 0, 1), np.float32), cv2.COLOR_RGB2GRAY)
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).reshape(-1)
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def fingerprint_photo(library: Library, photo_id: str) -> None:
    """Indexing stage: fingerprint one photo (as imported, before any edits)."""
    value = fingerprint(library.store.image(photo_id).srgb_proxy)
    library.db.run("UPDATE photos SET fingerprint = ? WHERE id = ?", (value, photo_id))
