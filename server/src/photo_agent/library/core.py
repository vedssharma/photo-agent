"""The library: a whole collection of photos, each one an ordinary document.

Photos join the library when they are imported (many at once, or a folder), and when one is
opened in the editor on its own. Every document already on disk joins the first time the
library is used. Removing a photo from the library deletes its document.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from photo_agent import imaging
from photo_agent.graph import Document
from photo_agent.library import faces, tagging
from photo_agent.library.db import LibraryDB, Row, placeholders
from photo_agent.library.filters import Flag, Label, PhotoFilter, order_by, where
from photo_agent.library.indexer import Indexer, IndexingMode
from photo_agent.render import RenderCache
from photo_agent.store import DocumentNotFoundError, DocumentStore
from photo_agent.vision.worker import ModelWorker

if TYPE_CHECKING:
    from photo_agent.library.batch import BatchStatus

log = logging.getLogger(__name__)

THUMBNAIL_LONG_EDGE = 480
THUMBNAIL_QUALITY = 82


class PhotoNotFoundError(KeyError):
    pass


class LibraryPhoto(BaseModel):
    """What the library shows of one photo."""

    id: str = Field(description="The photo's document id; open it in the editor with this.")
    filename: str
    format: str
    width: int
    height: int
    file_size: int
    imported_at: datetime
    revision: str
    """The document's current revision ("original" when unedited), for thumbnail URLs."""
    edited: bool
    rating: int = Field(0, description="Stars, 0 to 5.")
    flag: Flag = "none"
    label: Label | None = None
    albums: list[str] = Field(default_factory=list, description="Ids of albums it is in.")
    taken_at: datetime | None = Field(
        None, description="When it was taken, in the camera's local time, if known."
    )
    camera: str | None = None
    lens: str | None = None
    iso: int | None = None
    exposure_time: float | None = Field(None, description="Seconds.")
    f_number: float | None = None
    focal_length: float | None = Field(None, description="Millimetres.")
    latitude: float | None = None
    longitude: float | None = None
    place: str | None = Field(None, description="The place it was taken (see facets).")
    tags: list[str] = Field(default_factory=list, description="What is in it, best first.")
    people: list[str] = Field(default_factory=list, description="Ids of people in it.")


class Skipped(BaseModel):
    filename: str
    reason: str
    existing: str | None = Field(None, description="The library photo it is a copy of.")


class ImportResult(BaseModel):
    imported: list[LibraryPhoto]
    skipped: list[Skipped]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Library:
    def __init__(
        self,
        data_dir: Path,
        store: DocumentStore,
        worker: ModelWorker | None,
        renders: RenderCache,
        indexing: IndexingMode = "background",
    ) -> None:
        from photo_agent.library.stages import STAGES

        self.db = LibraryDB(data_dir / "library.db")
        self.store = store
        self.worker = worker
        self.renders = renders
        self.indexer = Indexer(self, STAGES, indexing)
        self.batches: dict[str, BatchStatus] = {}
        """Batch edits started since the server started, by id (see `batch.py`)."""
        store.listeners.append(self._saved)
        self._backfill()

    # Membership

    def has(self, photo_id: str) -> bool:
        return self.db.one("SELECT 1 FROM photos WHERE id = ?", (photo_id,)) is not None

    def copy_of(self, digest: str) -> str | None:
        """A library photo with exactly these bytes, if there is one."""
        row = self.db.one("SELECT id FROM photos WHERE sha256 = ? LIMIT 1", (digest,))
        return str(row["id"]) if row else None

    def import_file(self, filename: str, data: bytes) -> LibraryPhoto | Skipped:
        """Add one photo, unless the very same file is already in the library. Raises
        UnsupportedImageError. Call `indexer.kick()` once the batch is in."""
        existing = self.copy_of(sha256(data))
        if existing is not None:
            return Skipped(filename=filename, reason="Already in your library.", existing=existing)
        doc = self.store.create(filename, data)
        return self.add(doc, data)

    def add(self, doc: Document, data: bytes) -> LibraryPhoto:
        """Put a stored document in the library (no-op when it already is)."""
        if not self.has(doc.id):
            self.db.run(
                "INSERT INTO photos (id, filename, format, width, height, file_size, sha256, "
                "imported_at, revision) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    doc.id,
                    doc.filename,
                    doc.format,
                    doc.width,
                    doc.height,
                    len(data),
                    sha256(data),
                    datetime.now(UTC).isoformat(),
                    doc.revision,
                ),
            )
            self._write_thumbnail(doc.id)
        return self.get(doc.id)

    def remove(self, photo_ids: list[str]) -> int:
        """Delete photos from the library, and their documents with them."""
        known = [
            r["id"]
            for r in self.db.all(
                f"SELECT id FROM photos WHERE id IN ({placeholders(photo_ids)})", photo_ids
            )
        ]
        if not known:
            return 0
        self.db.run(f"DELETE FROM photos WHERE id IN ({placeholders(known)})", known)
        for photo_id in known:
            self.store.delete(photo_id)
        return len(known)

    def _backfill(self) -> None:
        """Bring documents made before the library existed (or outside it) in."""
        known = {r["id"] for r in self.db.all("SELECT id FROM photos")}
        added = False
        for doc_id in self.store.ids():
            if doc_id in known:
                continue
            try:
                doc = self.store.get(doc_id)
                data = self.store.original_file(doc_id).read_bytes()
            except (DocumentNotFoundError, ValueError, OSError):
                continue
            self.db.run(
                "INSERT INTO photos (id, filename, format, width, height, file_size, sha256, "
                "imported_at, revision) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    doc.id,
                    doc.filename,
                    doc.format,
                    doc.width,
                    doc.height,
                    len(data),
                    sha256(data),
                    doc.created_at.isoformat(),
                    doc.revision,
                ),
            )
            added = True
        if added:
            self.indexer.kick()

    def _saved(self, doc: Document) -> None:
        self.db.run("UPDATE photos SET revision = ? WHERE id = ?", (doc.revision, doc.id))

    def indexed(self, photo_id: str) -> None:
        """Called when a photo has been through every indexing stage."""

    # Reading

    def get(self, photo_id: str) -> LibraryPhoto:
        row = self.db.one("SELECT * FROM photos WHERE id = ?", (photo_id,))
        if row is None:
            raise PhotoNotFoundError(photo_id)
        return self._with_extras([photo_of(row)])[0]

    def photos(self, f: PhotoFilter | None = None) -> list[LibraryPhoto]:
        """The photos a filter lets through (all of them by default), newest import first."""
        f = f or PhotoFilter()
        clause, params = where(f)
        rows = self.db.all(
            f"SELECT p.* FROM photos p WHERE {clause} ORDER BY {order_by(f)}", params
        )
        return self._with_extras([photo_of(row) for row in rows])

    def _with_extras(self, photos: list[LibraryPhoto]) -> list[LibraryPhoto]:
        """Fill in what lives in other tables: albums, tags, and people."""
        member: dict[str, list[str]] = {}
        for r in self.db.all("SELECT album_id, photo_id FROM album_photos ORDER BY album_id"):
            member.setdefault(r["photo_id"], []).append(r["album_id"])
        tags = tagging.tags_of(self.db)
        people = faces.people_in(self.db)
        for photo in photos:
            photo.albums = member.get(photo.id, [])
            photo.tags = tags.get(photo.id, [])
            photo.people = people.get(photo.id, [])
        return photos

    # Thumbnails

    def thumbnail(self, photo_id: str) -> bytes:
        """A small JPEG of the photo as it looks now, edits included."""
        photo = self.get(photo_id)
        folder = self.store.folder(photo_id)
        if not photo.edited:
            path = folder / "thumb.jpg"
            if not path.is_file():
                self._write_thumbnail(photo_id)
            return path.read_bytes()
        path = folder / f"thumb-{photo.revision[:20]}.jpg"
        if not path.is_file():
            doc = self.store.get(photo_id)
            loaded = self.store.image(photo_id)
            pixels = self.renders.get_or_render(
                photo_id, loaded.proxy, doc.state, loaded.proxy_context
            )
            for old in folder.glob("thumb-*.jpg"):
                old.unlink(missing_ok=True)
            _write_jpeg(path, imaging.resize_long_edge(pixels, THUMBNAIL_LONG_EDGE))
        return path.read_bytes()

    def _write_thumbnail(self, photo_id: str) -> None:
        proxy = self.store.image(photo_id).proxy
        path = self.store.folder(photo_id) / "thumb.jpg"
        _write_jpeg(path, imaging.resize_long_edge(proxy, THUMBNAIL_LONG_EDGE))


def _write_jpeg(path: Path, pixels: imaging.Array) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(imaging.encode_jpeg(pixels, THUMBNAIL_QUALITY))
    tmp.replace(path)


def photo_of(row: Row) -> LibraryPhoto:
    return LibraryPhoto(
        id=row["id"],
        filename=row["filename"],
        format=row["format"],
        width=row["width"],
        height=row["height"],
        file_size=row["file_size"],
        imported_at=datetime.fromisoformat(row["imported_at"]),
        revision=row["revision"],
        edited=row["revision"] != "original",
        rating=row["rating"],
        flag=row["flag"],
        label=row["label"],
        taken_at=datetime.fromisoformat(row["taken_at"]) if row["taken_at"] else None,
        camera=row["camera"],
        lens=row["lens"],
        iso=row["iso"],
        exposure_time=row["exposure_time"],
        f_number=row["f_number"],
        focal_length=row["focal_length"],
        latitude=row["latitude"],
        longitude=row["longitude"],
        place=row["place"],
    )
