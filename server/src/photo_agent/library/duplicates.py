"""Duplicates: the same photo twice, and near duplicates (the same shot resized, re-saved,
lightly edited, or taken twice in a row).

Exact copies of a file never get in (import skips them), but the same photo can still arrive
as a different file: an exported copy, a screenshot of it, a resized version for sharing.
Each photo gets a fingerprint when indexed (see `fingerprints.py`). Copies of a photo have
the same or nearly the same fingerprint whatever their size, format, or compression; photos
that merely look alike (the same beach, another day) differ in many bits.

Photos whose fingerprints are close are grouped, and in each group one is suggested to keep:
the one someone has shown they care about (picked, rated, edited), else the largest.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, Field

from photo_agent.library.core import Library, LibraryPhoto
from photo_agent.library.db import LibraryDB

NEAR = 10
"""Fingerprints at most this many bits apart (of 64) are the same shot."""
EXACT = 2
"""At most this many apart: the same photo, only resized or re-saved."""


def _distances(values: npt.NDArray[np.uint64]) -> npt.NDArray[np.int64]:
    """Bits different between every pair of fingerprints."""
    xor = values[:, None] ^ values[None, :]
    as_bytes = xor.view(np.uint8).reshape(len(values), len(values), 8)
    return np.unpackbits(as_bytes, axis=2).sum(axis=2, dtype=np.int64)


class DuplicateGroup(BaseModel):
    kind: Literal["exact", "near"] = Field(
        description='"exact": the same photo (resized or re-saved at most); "near": the '
        "same shot, slightly different."
    )
    keep: str = Field(description="The photo suggested to keep.")
    photos: list[LibraryPhoto] = Field(description="The group, the one to keep first.")


class NotDuplicates(BaseModel):
    ids: list[str] = Field(min_length=2, max_length=200)


def groups(library: Library, ids: list[str] | None = None) -> list[DuplicateGroup]:
    """Groups of duplicates among these photos (all by default), biggest group first."""
    photos = {p.id: p for p in library.photos()}
    rows = [
        r
        for r in library.db.all(
            "SELECT id, fingerprint FROM photos WHERE fingerprint IS NOT NULL ORDER BY id"
        )
        if r["id"] in photos and (ids is None or r["id"] in ids)
    ]
    if len(rows) < 2:
        return []
    values = np.array([int(r["fingerprint"], 16) for r in rows], np.uint64)
    distance = _distances(values)
    apart = _not_duplicates(library.db)
    # Group by linking every close pair (union-find), except pairs said to differ.
    parent = list(range(len(rows)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    close = np.argwhere(np.triu(distance <= NEAR, k=1))
    for i, j in close:
        if (rows[i]["id"], rows[j]["id"]) not in apart:
            parent[root(int(i))] = root(int(j))
    members: dict[int, list[int]] = {}
    for i in range(len(rows)):
        members.setdefault(root(i), []).append(i)
    found = []
    for group in members.values():
        if len(group) < 2:
            continue
        ranked = sorted((photos[rows[i]["id"]] for i in group), key=_keeper)
        worst = max(int(distance[i, j]) for i in group for j in group)
        found.append(
            DuplicateGroup(
                kind="exact" if worst <= EXACT else "near",
                keep=ranked[0].id,
                photos=ranked,
            )
        )
    found.sort(key=lambda g: (-len(g.photos), g.keep))
    return found


def _keeper(photo: LibraryPhoto) -> tuple[object, ...]:
    """Sort key: the photo most worth keeping first."""
    return (
        photo.flag == "rejected",
        photo.flag != "picked",
        -photo.rating,
        not photo.edited,
        -(photo.width * photo.height),
        -photo.file_size,
        photo.imported_at,
        photo.id,
    )


def _not_duplicates(db: LibraryDB) -> set[tuple[str, str]]:
    pairs = set()
    for r in db.all("SELECT photo_a, photo_b FROM not_duplicates"):
        pairs.add((r["photo_a"], r["photo_b"]))
        pairs.add((r["photo_b"], r["photo_a"]))
    return pairs


def mark_not_duplicates(db: LibraryDB, ids: list[str]) -> None:
    """These photos are different photos, however alike: stop grouping them."""
    unique = sorted(set(ids))
    db.run_many(
        "INSERT OR IGNORE INTO not_duplicates (photo_a, photo_b) "
        "SELECT a.id, b.id FROM photos a, photos b WHERE a.id = ? AND b.id = ?",
        [(a, b) for i, a in enumerate(unique) for b in unique[i + 1 :]],
    )
