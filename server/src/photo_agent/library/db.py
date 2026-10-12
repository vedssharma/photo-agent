"""The library database: which photos are in the library, and what has been learned about them.

SQLite, in `<data_dir>/library.db`. A library photo is an ordinary document (see `store.py`):
its original and edits live in the document folder, and opening it in the editor works as
for any photo. The database holds what a whole collection needs to be sorted, filtered, and
searched quickly without opening every document: file facts, ratings, albums, metadata, tags,
faces, and the scores behind culling.

The schema grows by appending to MIGRATIONS; `PRAGMA user_version` records how far a
database has come. Postgres can take over when the app is productized: the queries here are
plain SQL with no SQLite-only features beyond JSON in text columns.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

MIGRATIONS: list[str] = [
    # 1: photos and the indexing stages each has been through.
    """
    CREATE TABLE photos (
        id TEXT PRIMARY KEY,
        filename TEXT NOT NULL,
        format TEXT NOT NULL,
        width INTEGER NOT NULL,
        height INTEGER NOT NULL,
        file_size INTEGER NOT NULL,
        sha256 TEXT NOT NULL,
        imported_at TEXT NOT NULL,
        revision TEXT NOT NULL DEFAULT 'original',
        index_key TEXT
    );
    CREATE INDEX photos_sha256 ON photos (sha256);
    CREATE INDEX photos_imported ON photos (imported_at, id);
    CREATE TABLE photo_stages (
        photo_id TEXT NOT NULL REFERENCES photos (id) ON DELETE CASCADE,
        stage TEXT NOT NULL,
        version INTEGER NOT NULL,
        error TEXT,
        PRIMARY KEY (photo_id, stage)
    );
    """,
]

Row = sqlite3.Row


class LibraryDB:
    """A thread-safe connection to the library database. Statements run one at a time,
    which SQLite wants anyway, and `transaction()` groups several into one."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._conn.execute("PRAGMA journal_mode = WAL")
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._migrate()

    def _migrate(self) -> None:
        version = int(self._conn.execute("PRAGMA user_version").fetchone()[0])
        for number, script in enumerate(MIGRATIONS[version:], start=version + 1):
            self._conn.executescript(f"BEGIN; {script}; PRAGMA user_version = {number}; COMMIT;")

    @property
    def version(self) -> int:
        return int(self.one("PRAGMA user_version")[0])  # type: ignore[index]

    def all(self, sql: str, params: Sequence[Any] = ()) -> list[Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def one(self, sql: str, params: Sequence[Any] = ()) -> Row | None:
        with self._lock:
            row: Row | None = self._conn.execute(sql, params).fetchone()
            return row

    def run(self, sql: str, params: Sequence[Any] = ()) -> int:
        """Run a statement; returns how many rows it changed."""
        with self._lock:
            return self._conn.execute(sql, params).rowcount

    def run_many(self, sql: str, rows: Sequence[Sequence[Any]]) -> None:
        with self.transaction():
            self._conn.executemany(sql, rows)

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Run the statements inside as one, all or nothing. Nests (inner ones join)."""
        with self._lock:
            if self._conn.in_transaction:
                yield
                return
            self._conn.execute("BEGIN")
            try:
                yield
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            self._conn.execute("COMMIT")

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def placeholders(values: Sequence[Any]) -> str:
    """`?, ?, ?` for an IN (...) list."""
    return ", ".join("?" for _ in values)
