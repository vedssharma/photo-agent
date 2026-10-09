"""A small on-disk cache of Claude's analyses of a photo (suggestions, critiques).

An analysis depends only on what the photo looks like (the document's revision) and who
made it (the model, or the built-in fallback), so asking again for the same edits, or
after undoing back to them, reuses the answer instead of paying for another look. Kept
next to the document:

    <data_dir>/documents/<id>/analyses/<kind>-<key>.json
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

KEEP = 12
"""Analyses kept per kind and document; the oldest go first."""

T = TypeVar("T", bound=BaseModel)


class AnalysisCache:
    def __init__(self, doc_folder: Path) -> None:
        self.folder = doc_folder / "analyses"

    def _path(self, kind: str, revision: str, source: str) -> Path:
        key = hashlib.sha256(f"{revision}\0{source}".encode()).hexdigest()[:20]
        return self.folder / f"{kind}-{key}.json"

    def get(self, kind: str, revision: str, source: str, model: type[T]) -> T | None:
        path = self._path(kind, revision, source)
        if not path.is_file():
            return None
        try:
            return model.model_validate_json(path.read_text())
        except ValueError:
            return None

    def put(self, kind: str, revision: str, source: str, value: BaseModel) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        path = self._path(kind, revision, source)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(value.model_dump_json())
        tmp.replace(path)
        older = sorted(self.folder.glob(f"{kind}-*.json"), key=lambda p: p.stat().st_mtime)
        for stale in older[:-KEEP]:
            stale.unlink(missing_ok=True)
