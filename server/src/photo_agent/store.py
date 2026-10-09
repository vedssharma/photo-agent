"""Local-disk storage for documents: one folder per document with the original and its graph.

    <data_dir>/documents/<id>/original.<ext>   the uploaded bytes, never modified
    <data_dir>/documents/<id>/document.json    the edit graph
    <data_dir>/documents/<id>/vision/          cached model results (semantic masks, ...)
    <data_dir>/documents/<id>/suggestions.json the edit directions offered on open
    <data_dir>/documents/<id>/references/      photos shared to match (see references.py)

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
from photo_agent.graph import Document, new_id
from photo_agent.render import RenderContext
from photo_agent.vision.backends import BackendMode
from photo_agent.vision.selection import DocumentVision
from photo_agent.vision.tasks import TASKS
from photo_agent.vision.worker import ModelWorker

EXTENSIONS = {"JPEG": ".jpg", "PNG": ".png", "HEIF": ".heic"}
_ID_RE = re.compile(r"^[0-9a-f]{12}$")


class DocumentNotFoundError(KeyError):
    pass


class MismatchError(ValueError):
    pass


@dataclass
class LoadedImage:
    source: imaging.DecodedImage
    proxy: imaging.Array
    vision: DocumentVision | None = None
    """Runs the models behind semantic masks; None when the store has no model worker."""

    @property
    def proxy_scale(self) -> float:
        src_h, src_w = self.source.pixels.shape[:2]
        return float(max(self.proxy.shape[:2]) / max(src_h, src_w))

    @property
    def proxy_context(self) -> RenderContext:
        """How to render the proxy so it matches a full-resolution render."""
        return RenderContext(
            scale=self.proxy_scale,
            source_aspect=self.source.width / self.source.height,
            vision=self.vision,
        )

    @property
    def full_context(self) -> RenderContext:
        return RenderContext(
            scale=1.0, source_aspect=self.source.width / self.source.height, vision=self.vision
        )


class DocumentStore:
    def __init__(
        self,
        root: Path,
        cache_size: int = 4,
        worker: ModelWorker | None = None,
        backends: BackendMode = "auto",
    ) -> None:
        self.data_dir = root
        self.root = root / "documents"
        self.worker = worker
        self.backends = backends
        self._cache: OrderedDict[str, LoadedImage] = OrderedDict()
        self._cache_size = cache_size
        self._lock = threading.Lock()

    def create(self, filename: str, data: bytes, edits: Document | None = None) -> Document:
        """Decode an upload and store it as a new document, optionally with existing edits
        (when reopening a project). Raises UnsupportedImageError, or MismatchError when the
        edits were made on a photo of a different size."""
        decoded = imaging.decode(data)
        if edits is None:
            doc = Document(
                filename=_safe_filename(filename),
                format=decoded.format,
                width=decoded.width,
                height=decoded.height,
            )
        else:
            if (edits.width, edits.height) != (decoded.width, decoded.height):
                raise MismatchError("The photo does not match the edits saved with it.")
            doc = edits.model_copy(
                update={"filename": _safe_filename(edits.filename), "format": decoded.format}
            )
            # Keep the id so links and browser autosaves still find it, unless it is taken.
            if not _ID_RE.match(doc.id) or self.exists(doc.id):
                doc.id = new_id()
        folder = self._folder(doc.id)
        folder.mkdir(parents=True)
        (folder / f"original{EXTENSIONS[decoded.format]}").write_bytes(data)
        self.save(doc)
        self._remember(doc.id, self._loaded(doc.id, decoded))
        return doc

    def model_for(self, task: str) -> str:
        """The backend that runs a model task here (a model, or "classical")."""
        choices = TASKS[task].choices(self.backends)
        return choices[0].name if choices else "none"

    def exists(self, doc_id: str) -> bool:
        try:
            return (self._folder(doc_id) / "document.json").is_file()
        except DocumentNotFoundError:
            return False

    def original_file(self, doc_id: str) -> Path:
        """The uploaded file, untouched."""
        folder = self._folder(doc_id)
        originals = sorted(folder.glob("original.*")) if folder.is_dir() else []
        if not originals:
            raise DocumentNotFoundError(doc_id)
        return originals[0]

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

    def folder(self, doc_id: str) -> Path:
        """The folder holding a document's files."""
        return self._folder(doc_id)

    def file(self, doc_id: str, name: str) -> Path:
        """A file kept next to a document (suggestions, references, ...)."""
        return self._folder(doc_id) / name

    def image(self, doc_id: str) -> LoadedImage:
        """The decoded original and its preview proxy."""
        with self._lock:
            cached = self._cache.get(doc_id)
            if cached is not None:
                self._cache.move_to_end(doc_id)
                return cached
        decoded = imaging.decode(self.original_file(doc_id).read_bytes())
        loaded = self._loaded(doc_id, decoded)
        self._remember(doc_id, loaded)
        return loaded

    def _loaded(self, doc_id: str, decoded: imaging.DecodedImage) -> LoadedImage:
        loaded = LoadedImage(decoded, imaging.make_proxy(decoded.pixels))
        if self.worker is not None:
            loaded.vision = DocumentVision(
                doc_id,
                loaded.proxy,
                source_aspect=decoded.width / decoded.height,
                scale=loaded.proxy_scale,
                worker=self.worker,
                folder=self._folder(doc_id) / "vision",
                backends=self.backends,
            )
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
