"""Local-disk storage for documents: one folder per document with the original and its graph.

    <data_dir>/documents/<id>/original.<ext>   the uploaded bytes, never modified
    <data_dir>/documents/<id>/document.json    the edit graph

Decoded originals and preview proxies are kept in a small in-memory cache so a chat turn
does not re-decode the file on every render.
"""

from __future__ import annotations

import re
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from photo_agent import imaging
from photo_agent.graph import Document

EXTENSIONS = {"JPEG": ".jpg", "PNG": ".png", "HEIF": ".heic"}
_ID_RE = re.compile(r"^[0-9a-f]{12}$")


class DocumentNotFoundError(KeyError):
    pass


@dataclass
class LoadedImage:
    source: imaging.DecodedImage
    proxy: imaging.Array


class DocumentStore:
    def __init__(self, root: Path, cache_size: int = 4) -> None:
        self.root = root / "documents"
        self._cache: OrderedDict[str, LoadedImage] = OrderedDict()
        self._cache_size = cache_size
        self._lock = threading.Lock()

    def create(self, filename: str, data: bytes) -> Document:
        """Decode an upload and store it as a new document. Raises UnsupportedImageError."""
        decoded = imaging.decode(data)
        doc = Document(
            filename=_safe_filename(filename),
            format=decoded.format,
            width=decoded.width,
            height=decoded.height,
        )
        folder = self._folder(doc.id)
        folder.mkdir(parents=True)
        (folder / f"original{EXTENSIONS[decoded.format]}").write_bytes(data)
        self.save(doc)
        self._remember(doc.id, LoadedImage(decoded, imaging.make_proxy(decoded.pixels)))
        return doc

    def get(self, doc_id: str) -> Document:
        path = self._folder(doc_id) / "document.json"
        if not path.is_file():
            raise DocumentNotFoundError(doc_id)
        return Document.model_validate_json(path.read_text())

    def save(self, doc: Document) -> None:
        path = self._folder(doc.id) / "document.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(doc.model_dump_json(indent=2))
        tmp.replace(path)

    def image(self, doc_id: str) -> LoadedImage:
        """The decoded original and its preview proxy."""
        with self._lock:
            cached = self._cache.get(doc_id)
            if cached is not None:
                self._cache.move_to_end(doc_id)
                return cached
        folder = self._folder(doc_id)
        originals = sorted(folder.glob("original.*")) if folder.is_dir() else []
        if not originals:
            raise DocumentNotFoundError(doc_id)
        decoded = imaging.decode(originals[0].read_bytes())
        loaded = LoadedImage(decoded, imaging.make_proxy(decoded.pixels))
        self._remember(doc_id, loaded)
        return loaded

    def _remember(self, doc_id: str, loaded: LoadedImage) -> None:
        with self._lock:
            self._cache[doc_id] = loaded
            self._cache.move_to_end(doc_id)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)

    def _folder(self, doc_id: str) -> Path:
        if not _ID_RE.match(doc_id):
            raise DocumentNotFoundError(doc_id)
        return self.root / doc_id


def _safe_filename(name: str) -> str:
    """Keep only the base name, for display and for naming exports."""
    base = Path(name.replace("\\", "/")).name.strip()
    return base[:200] or "photo"
