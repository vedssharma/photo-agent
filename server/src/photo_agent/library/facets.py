"""What there is to filter by: the cameras, places, and years in the library."""

from __future__ import annotations

from pydantic import BaseModel

from photo_agent.library.db import LibraryDB
from photo_agent.library.metadata import place_label


class Facet(BaseModel):
    value: str
    """What to filter by (a camera name, a place id, a year)."""
    label: str
    count: int


class Facets(BaseModel):
    cameras: list[Facet]
    places: list[Facet]
    years: list[Facet]
    """Years photos were taken, newest first."""


def facets(db: LibraryDB) -> Facets:
    cameras = db.all(
        "SELECT camera, COUNT(*) AS n FROM photos WHERE camera IS NOT NULL "
        "GROUP BY camera ORDER BY n DESC, camera"
    )
    places = db.all(
        "SELECT place, COUNT(*) AS n FROM photos WHERE place IS NOT NULL "
        "GROUP BY place ORDER BY n DESC, place"
    )
    years = db.all(
        "SELECT substr(taken_at, 1, 4) AS year, COUNT(*) AS n FROM photos "
        "WHERE taken_at IS NOT NULL GROUP BY year ORDER BY year DESC"
    )
    return Facets(
        cameras=[Facet(value=r["camera"], label=r["camera"], count=r["n"]) for r in cameras],
        places=[
            Facet(value=r["place"], label=place_label(r["place"]), count=r["n"]) for r in places
        ],
        years=[Facet(value=r["year"], label=r["year"], count=r["n"]) for r in years],
    )
