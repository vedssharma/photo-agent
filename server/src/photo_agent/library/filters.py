"""Narrowing the library down: by album, rating, flag, and color label."""

from __future__ import annotations

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
    return (" AND ".join(clauses) or "1"), params
