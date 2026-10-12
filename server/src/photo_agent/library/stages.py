"""The indexing stages every library photo goes through, in order (see `indexer.py`)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from photo_agent.library import faces, fingerprints, metadata, tagging
from photo_agent.library.indexer import Stage
from photo_agent.vision import quality

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


def assess(library: Library, photo_id: str) -> None:
    """How good a shot it is: sharpness, exposure, appeal, and closed eyes."""
    found = quality.measure(library.store.image(photo_id).proxy)
    library.db.run(
        "UPDATE photos SET sharpness = ?, exposure = ?, appeal = ?, faces = ?, eyes_closed = ? "
        "WHERE id = ?",
        (found.sharpness, found.exposure, found.appeal, found.faces, found.eyes_closed, photo_id),
    )


def _has_worker(library: Library) -> bool:
    return library.worker is not None


def _faces_on(library: Library) -> bool:
    return library.worker is not None and faces.enabled(library.db)


STAGES: list[Stage] = [
    Stage("thumbnail", 1, "Making a thumbnail", _thumbnail),
    Stage("metadata", 1, "Reading photo details", _metadata),
    Stage("fingerprint", 1, "Looking for duplicates", fingerprints.fingerprint_photo),
    Stage("quality", 1, "Judging sharpness and exposure", assess),
    Stage("tags", 1, "Tagging", tagging.tag_photo, _has_worker),
    Stage("faces", 1, "Finding faces", faces.find_faces, _faces_on),
]
