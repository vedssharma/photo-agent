"""Narrowing the library down: by album, rating, flag, label, date, camera, place, and tags."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Literal

from pydantic import BaseModel, Field

Flag = Literal["picked", "rejected", "none"]
Label = Literal["red", "yellow", "green", "blue", "purple"]


class PhotoFilter(BaseModel):
    """Which photos to show. Every condition given must hold."""

    album: str | None = Field(None, description="Only photos in this album.")
    min_rating: int = Field(0, ge=0, le=5, description="Only photos with at least this many stars.")
    flag: Flag | None = Field(None, description="Only picked, rejected, or unflagged photos.")
    label: Label | None = Field(None, description="Only photos with this color label.")
    taken_after: date | None = Field(None, description="Only photos taken on or after this day.")
    taken_before: date | None = Field(None, description="Only photos taken on or before this day.")
    camera: str | None = Field(None, description="Only photos from this camera (see facets).")
    place: str | None = Field(None, description="Only photos from this place (see facets).")
    has_location: bool | None = Field(None, description="Only photos with (or without) GPS.")
    tags: list[str] = Field(
        default_factory=list, max_length=10, description="Only photos with every one of these tags."
    )
    sort: Literal["taken", "imported"] = Field(
        "taken", description="Newest first by when they were taken (or imported)."
    )


def where(f: PhotoFilter) -> tuple[str, list[Any]]:
    """A WHERE clause (on `photos p`) and its parameters for a filter."""
    clauses: list[str] = []
    params: list[Any] = []
    if f.album:
        clauses.append("p.id IN (SELECT photo_id FROM album_photos WHERE album_id = ?)")
        params.append(f.album)
    if f.min_rating:
        clauses.append("p.rating >= ?")
        params.append(f.min_rating)
    if f.flag:
        clauses.append("p.flag = ?")
        params.append(f.flag)
    if f.label:
        clauses.append("p.label = ?")
        params.append(f.label)
    if f.taken_after:
        clauses.append("p.taken_at >= ?")
        params.append(f.taken_after.isoformat())
    if f.taken_before:
        clauses.append("p.taken_at < ?")
        params.append((f.taken_before + timedelta(days=1)).isoformat())
    if f.camera:
        clauses.append("p.camera = ?")
        params.append(f.camera)
    if f.place:
        clauses.append("p.place = ?")
        params.append(f.place)
    if f.tags:
        clauses.append(
            "p.id IN (SELECT photo_id FROM tags WHERE source != 'removed' "
            f"AND tag IN ({', '.join('?' for _ in f.tags)}) "
            "GROUP BY photo_id HAVING COUNT(DISTINCT tag) = ?)"
        )
        params.extend([*f.tags, len(set(f.tags))])
    if f.has_location is not None:
        clauses.append("p.place IS NOT NULL" if f.has_location else "p.place IS NULL")
    return (" AND ".join(clauses) or "1"), params


def order_by(f: PhotoFilter) -> str:
    if f.sort == "imported":
        return "p.imported_at DESC, p.id DESC"
    # Photos without a capture time go by when they were imported.
    return "COALESCE(p.taken_at, p.imported_at) DESC, p.id DESC"
