"""Reference photos: other photos the person shares so theirs can be made to look alike.

A reference is kept small (it is only looked at and measured) next to the document:

    <data_dir>/documents/<id>/references/index.json   id, file name, and measurements
    <data_dir>/documents/<id>/references/<ref>.jpg    the reference, long edge 768 px

Its color and tone are measured once, when it is shared, and copied into each
`match_reference` operation, so the operation renders without the reference file (and
carries over in recipes to other photos).
"""

from __future__ import annotations

import threading
import uuid
from pathlib import Path

from pydantic import BaseModel, Field, TypeAdapter

from photo_agent import imaging
from photo_agent.layers import EditState
from photo_agent.operations import MatchReference, ReferenceStats
from photo_agent.render import reference_stats

LONG_EDGE = 768
MAX_REFERENCES = 50


class ReferenceNotFoundError(KeyError):
    pass


class Reference(BaseModel):
    id: str
    filename: str = Field(description="The shared file's name, for display.")
    stats: ReferenceStats


_INDEX: TypeAdapter[list[Reference]] = TypeAdapter(list[Reference])
_lock = threading.Lock()


def _folder(doc_folder: Path) -> Path:
    return doc_folder / "references"


def _read(doc_folder: Path) -> list[Reference]:
    path = _folder(doc_folder) / "index.json"
    if not path.is_file():
        return []
    try:
        return _INDEX.validate_json(path.read_text())
    except ValueError:
        return []


def add(doc_folder: Path, filename: str, data: bytes) -> Reference:
    """Decode, shrink, measure, and keep a shared photo. Raises UnsupportedImageError."""
    pixels = imaging.resize_long_edge(imaging.decode(data).pixels, LONG_EDGE)
    name = Path(filename.replace("\\", "/")).name.strip()[:120] or "reference"
    found = Reference(id="r" + uuid.uuid4().hex[:7], filename=name, stats=reference_stats(pixels))
    folder = _folder(doc_folder)
    with _lock:
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{found.id}.jpg").write_bytes(imaging.encode_jpeg(pixels, 88))
        kept = [*_read(doc_folder), found][-MAX_REFERENCES:]
        (folder / "index.json").write_bytes(_INDEX.dump_json(kept))
    return found


def get(doc_folder: Path, ref_id: str) -> Reference:
    for found in _read(doc_folder):
        if found.id == ref_id:
            return found
    raise ReferenceNotFoundError(ref_id)


def image_file(doc_folder: Path, ref_id: str) -> Path:
    get(doc_folder, ref_id)
    return _folder(doc_folder) / f"{ref_id}.jpg"


def pixels(doc_folder: Path, ref_id: str) -> imaging.Array:
    return imaging.decode(image_file(doc_folder, ref_id).read_bytes()).pixels


def fill_stats(state: EditState, doc_folder: Path) -> EditState:
    """The state with measurements filled into any match_reference operation lacking them
    (one added by hand from its id). Raises ReferenceNotFoundError for an unknown id."""
    if not any(
        isinstance(op, MatchReference) and op.stats is None for op in state.all_operations()
    ):
        return state
    state = state.model_copy(deep=True)
    for layer in state.layers:
        for i, op in enumerate(layer.operations):
            if isinstance(op, MatchReference) and op.stats is None:
                stats = get(doc_folder, op.reference).stats
                layer.operations[i] = op.model_copy(update={"stats": stats})
    return state
