"""Organizing the library by hand: star ratings, pick/reject flags, color labels, albums."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from photo_agent.library.db import LibraryDB, placeholders
from photo_agent.library.filters import Flag, Label


class AlbumNotFoundError(KeyError):
    pass


class Marks(BaseModel):
    """Changes to apply to several photos at once; leave out what should stay as it is."""

    ids: list[str] = Field(min_length=1, max_length=5000)
    rating: int | None = Field(None, ge=0, le=5, description="Stars, 0 to clear.")
    flag: Flag | None = None
    label: Label | Literal["none"] | None = Field(
        None, description='A color label, or "none" to clear it.'
    )


def mark(db: LibraryDB, marks: Marks) -> int:
    """Rate, flag, or label photos. Returns how many photos it changed."""
    sets: list[str] = []
    params: list[object] = []
    if marks.rating is not None:
        sets.append("rating = ?")
        params.append(marks.rating)
    if marks.flag is not None:
        sets.append("flag = ?")
        params.append(marks.flag)
    if marks.label is not None:
        sets.append("label = ?")
        params.append(None if marks.label == "none" else marks.label)
    if not sets:
        return 0
    return db.run(
        f"UPDATE photos SET {', '.join(sets)} WHERE id IN ({placeholders(marks.ids)})",
        [*params, *marks.ids],
    )


class Album(BaseModel):
    id: str
    name: str
    created_at: datetime
    count: int = Field(description="Photos in the album.")
    cover: str | None = Field(None, description="A photo to show for the album.")


class NewAlbum(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    name: str = Field(min_length=1, max_length=80)
    photo_ids: list[str] = Field(default_factory=list, max_length=5000)


class AlbumChange(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    name: str = Field(min_length=1, max_length=80)


def albums(db: LibraryDB) -> list[Album]:
    """Every album, by name, with how many photos it has and its newest one as cover."""
    rows = db.all(
        "SELECT a.id, a.name, a.created_at, COUNT(ap.photo_id) AS count, "
        "(SELECT photo_id FROM album_photos WHERE album_id = a.id "
        " ORDER BY added_at DESC, photo_id LIMIT 1) AS cover "
        "FROM albums a LEFT JOIN album_photos ap ON ap.album_id = a.id "
        "GROUP BY a.id ORDER BY a.name COLLATE NOCASE, a.id"
    )
    return [
        Album(
            id=r["id"],
            name=r["name"],
            created_at=datetime.fromisoformat(r["created_at"]),
            count=r["count"],
            cover=r["cover"],
        )
        for r in rows
    ]


def album(db: LibraryDB, album_id: str) -> Album:
    for found in albums(db):
        if found.id == album_id:
            return found
    raise AlbumNotFoundError(album_id)


def create_album(db: LibraryDB, new: NewAlbum) -> Album:
    album_id = uuid.uuid4().hex[:12]
    with db.transaction():
        db.run(
            "INSERT INTO albums (id, name, created_at) VALUES (?, ?, ?)",
            (album_id, new.name, datetime.now(UTC).isoformat()),
        )
        add_to_album(db, album_id, new.photo_ids)
    return album(db, album_id)


def rename_album(db: LibraryDB, album_id: str, change: AlbumChange) -> Album:
    if not db.run("UPDATE albums SET name = ? WHERE id = ?", (change.name, album_id)):
        raise AlbumNotFoundError(album_id)
    return album(db, album_id)


def delete_album(db: LibraryDB, album_id: str) -> None:
    """Delete an album; its photos stay in the library."""
    if not db.run("DELETE FROM albums WHERE id = ?", (album_id,)):
        raise AlbumNotFoundError(album_id)


def add_to_album(db: LibraryDB, album_id: str, photo_ids: list[str]) -> None:
    if db.one("SELECT 1 FROM albums WHERE id = ?", (album_id,)) is None:
        raise AlbumNotFoundError(album_id)
    if not photo_ids:
        return
    now = datetime.now(UTC).isoformat()
    db.run(
        "INSERT OR IGNORE INTO album_photos (album_id, photo_id, added_at) "
        f"SELECT ?, id, ? FROM photos WHERE id IN ({placeholders(photo_ids)})",
        [album_id, now, *photo_ids],
    )


def remove_from_album(db: LibraryDB, album_id: str, photo_ids: list[str]) -> None:
    if db.one("SELECT 1 FROM albums WHERE id = ?", (album_id,)) is None:
        raise AlbumNotFoundError(album_id)
    db.run(
        f"DELETE FROM album_photos WHERE album_id = ? AND photo_id IN ({placeholders(photo_ids)})",
        [album_id, *photo_ids],
    )
