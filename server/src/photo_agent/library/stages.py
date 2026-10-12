"""The indexing stages every library photo goes through, in order (see `indexer.py`)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from photo_agent.library import metadata, tagging
from photo_agent.library.indexer import Stage

if TYPE_CHECKING:
    from photo_agent.library.core import Library


def _thumbnail(library: Library, photo_id: str) -> None:
    if not (library.store.folder(photo_id) / "thumb.jpg").is_file():
        library._write_thumbnail(photo_id)


def _metadata(library: Library, photo_id: str) -> None:
    found = metadata.read(library.store.original_file(photo_id).read_bytes())
    library.db.run(
        "UPDATE photos SET taken_at = ?, camera = ?, lens = ?, iso = ?, exposure_time = ?, "
        "f_number = ?, focal_length = ?, latitude = ?, longitude = ?, place = ? WHERE id = ?",
        (
            found.taken_at.isoformat() if found.taken_at else None,
            found.camera,
            found.lens,
            found.iso,
            found.exposure_time,
            found.f_number,
            found.focal_length,
            found.latitude,
            found.longitude,
            found.place,
            photo_id,
        ),
    )


def _has_worker(library: Library) -> bool:
    return library.worker is not None


STAGES: list[Stage] = [
    Stage("thumbnail", 1, "Making a thumbnail", _thumbnail),
    Stage("metadata", 1, "Reading photo details", _metadata),
    Stage("tags", 1, "Tagging", tagging.tag_photo, _has_worker),
]
