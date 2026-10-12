"""The indexing stages every library photo goes through, in order (see `indexer.py`)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from photo_agent.library.indexer import Stage

if TYPE_CHECKING:
    from photo_agent.library.core import Library


def _thumbnail(library: Library, photo_id: str) -> None:
    if not (library.store.folder(photo_id) / "thumb.jpg").is_file():
        library._write_thumbnail(photo_id)


STAGES: list[Stage] = [
    Stage("thumbnail", 1, "Making a thumbnail", _thumbnail),
]
