"""Auto-tagging: what is in each photo, found by a vision model when indexing.

The model worker's `tag_photo` task returns tags with scores and an embedding of the photo:
SigLIP when the models are installed, else the classical stand-in (`vision/library.py`).
Tags anyone adds by hand are kept apart from found ones, and a found tag someone removes
stays removed when the photo is tagged again.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field

from photo_agent import imaging
from photo_agent.library.db import LibraryDB, placeholders

if TYPE_CHECKING:
    from photo_agent.library.core import Library

TAG_EDGE = 448
"""Long edge of the copy the model looks at."""

Source = Literal["auto", "manual", "removed"]


def tag_photo(library: Library, photo_id: str) -> None:
    """Indexing stage: find tags and an embedding for one photo."""
    assert library.worker is not None
    pixels = imaging.resize_long_edge(library.store.image(photo_id).srgb_proxy, TAG_EDGE)
    # Not tied to the document: indexing is not something its editor waits on.
    result = library.worker.run("tag_photo", pixels)
    found: dict[str, float] = result.value["tags"]
    vector = np.asarray(result.value["embedding"], np.float32)
    with library.db.transaction():
        library.db.run("DELETE FROM tags WHERE photo_id = ? AND source = 'auto'", (photo_id,))
        library.db.run_many(
            "INSERT OR IGNORE INTO tags (photo_id, tag, score, source) VALUES (?, ?, ?, 'auto')",
            [(photo_id, tag, round(score, 4)) for tag, score in found.items()],
        )
        library.db.run(
            "INSERT INTO embeddings (photo_id, model, vector) VALUES (?, ?, ?) "
            "ON CONFLICT (photo_id) DO UPDATE SET model = excluded.model, vector = excluded.vector",
            (photo_id, result.backend, vector.tobytes()),
        )


def tags_of(db: LibraryDB) -> dict[str, list[str]]:
    """Every photo's tags (found and added, not removed), best first."""
    out: dict[str, list[str]] = {}
    for r in db.all(
        "SELECT photo_id, tag FROM tags WHERE source != 'removed' "
        "ORDER BY photo_id, source = 'manual' DESC, score DESC, tag"
    ):
        out.setdefault(r["photo_id"], []).append(r["tag"])
    return out


def all_tags(db: LibraryDB) -> list[tuple[str, int]]:
    """Each tag in use and how many photos have it, most used first."""
    rows = db.all(
        "SELECT tag, COUNT(*) AS n FROM tags WHERE source != 'removed' "
        "GROUP BY tag ORDER BY n DESC, tag"
    )
    return [(r["tag"], r["n"]) for r in rows]


class TagChange(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    ids: list[str] = Field(min_length=1, max_length=5000)
    add: list[str] = Field(default_factory=list, max_length=20)
    remove: list[str] = Field(default_factory=list, max_length=20)


def change_tags(db: LibraryDB, change: TagChange) -> None:
    """Add or remove tags by hand on several photos."""
    add = [t.lower()[:40] for t in change.add if t.strip()]
    remove = [t.lower() for t in change.remove if t.strip()]
    with db.transaction():
        for tag in add:
            db.run(
                "INSERT INTO tags (photo_id, tag, score, source) "
                f"SELECT id, ?, 1.0, 'manual' FROM photos WHERE id IN ({placeholders(change.ids)}) "
                "ON CONFLICT (photo_id, tag) DO UPDATE SET source = 'manual', score = 1.0",
                [tag, *change.ids],
            )
        for tag in remove:
            # Remember the removal, so tagging the photo again does not bring it back.
            db.run(
                "UPDATE tags SET source = 'removed', score = 0 WHERE tag = ? "
                f"AND photo_id IN ({placeholders(change.ids)})",
                [tag, *change.ids],
            )


def vectors(db: LibraryDB, model: str, ids: list[str]) -> dict[str, npt.NDArray[np.float32]]:
    """Embeddings made by `model` for these photos."""
    if not ids:
        return {}
    rows = db.all(
        f"SELECT photo_id, vector FROM embeddings WHERE model = ? "
        f"AND photo_id IN ({placeholders(ids)})",
        [model, *ids],
    )
    return {r["photo_id"]: np.frombuffer(r["vector"], np.float32) for r in rows}
