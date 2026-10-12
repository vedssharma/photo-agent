"""Background indexing: learn about each library photo once, off the request path.

Indexing a photo runs a list of stages (a thumbnail, its metadata, tags, ...), each with a
version. A photo is indexed when it has been through every enabled stage at its current
version; bumping a stage's version, or enabling one (face clustering is opt-in), re-runs just
that stage on every photo. Progress survives restarts because it lives in the database.

Photos are indexed one at a time, oldest import first, all stages of a photo together so it
is decoded once. Stages that need a model submit a job to the model worker, so the heavy
lifting happens in that process and shows up in its job list; the indexing thread mostly
waits. A stage that fails is recorded with its error and not retried until its version
changes, so one odd file never stalls the queue.

In "inline" mode (tests) there is no thread: `kick()` indexes everything before returning.
"""

from __future__ import annotations

import logging
import threading
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

from photo_agent.library.db import LibraryDB, placeholders

if TYPE_CHECKING:
    from photo_agent.library.core import Library

log = logging.getLogger(__name__)

IndexingMode = Literal["background", "inline"]


@dataclass(frozen=True)
class Stage:
    name: str
    version: int
    label: str
    """What the person sees while it runs, e.g. "Reading photo details"."""
    run: Callable[[Library, str], None]
    enabled: Callable[[Library], bool] = lambda library: True


class IndexStatus(BaseModel):
    total: int
    """Photos in the library."""
    pending: int
    """Photos still waiting to be indexed (or being indexed now)."""
    current: str | None = None
    """What is happening now, e.g. "Tagging beach.jpg"."""


class Indexer:
    def __init__(self, library: Library, stages: list[Stage], mode: IndexingMode) -> None:
        self.library = library
        self.stages = stages
        self.mode = mode
        self._wake = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stopped = False
        self._current: str | None = None

    @property
    def db(self) -> LibraryDB:
        return self.library.db

    def enabled(self) -> list[Stage]:
        return [stage for stage in self.stages if stage.enabled(self.library)]

    def key(self) -> str:
        """Identifies the stages (and versions) a fully indexed photo has been through."""
        return ",".join(f"{s.name}:{s.version}" for s in self.enabled())

    def status(self) -> IndexStatus:
        key = self.key()
        row = self.db.one(
            "SELECT COUNT(*), SUM(index_key IS NOT ?) FROM photos",
            (key,),
        )
        total, pending = (int(row[0]), int(row[1] or 0)) if row else (0, 0)
        return IndexStatus(total=total, pending=pending, current=self._current if pending else None)

    def kick(self) -> None:
        """There may be new work: index it (inline) or wake the thread (background)."""
        if self.mode == "inline":
            with self._lock:
                while self._index_next():
                    pass
            return
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(
                    target=self._loop, name="photo-agent-indexer", daemon=True
                )
                self._thread.start()
        self._wake.set()

    def stop(self) -> None:
        self._stopped = True
        self._wake.set()

    def _loop(self) -> None:
        while not self._stopped:
            self._wake.clear()
            try:
                busy = self._index_next()
            except Exception:  # Keep the thread alive; the database may be briefly locked.
                log.exception("Indexing failed")
                busy = False
            if not busy:
                self._wake.wait(timeout=30)

    def _index_next(self) -> bool:
        """Index the next photo that needs it. Returns False when there is none."""
        key = self.key()
        row = self.db.one(
            "SELECT id, filename FROM photos WHERE index_key IS NOT ? "
            "ORDER BY imported_at, id LIMIT 1",
            (key,),
        )
        if row is None:
            self._current = None
            return False
        photo_id, filename = str(row["id"]), str(row["filename"])
        stages = self.enabled()
        done = {
            (r["stage"], r["version"])
            for r in self.db.all(
                f"SELECT stage, version FROM photo_stages WHERE photo_id = ? "
                f"AND stage IN ({placeholders(stages)})",
                (photo_id, *[s.name for s in stages]),
            )
        }
        for stage in stages:
            if (stage.name, stage.version) in done:
                continue
            self._current = f"{stage.label}: {filename}"
            error = None
            try:
                stage.run(self.library, photo_id)
            except Exception as exc:
                if not self.library.has(photo_id):
                    return True  # removed while it was being indexed
                error = str(exc) or type(exc).__name__
                log.warning("Indexing %s (%s) failed: %s", photo_id, stage.name, error)
                log.debug("%s", traceback.format_exc())
            self.db.run(
                "INSERT INTO photo_stages (photo_id, stage, version, error) "
                "SELECT id, ?, ?, ? FROM photos WHERE id = ? "
                "ON CONFLICT (photo_id, stage) DO UPDATE SET "
                "version = excluded.version, error = excluded.error",
                (stage.name, stage.version, error, photo_id),
            )
        self.db.run("UPDATE photos SET index_key = ? WHERE id = ?", (key, photo_id))
        self.library.indexed(photo_id)
        return True
