"""Natural-language search: "beach photos from last summer with the dog".

A query is first understood as filters the library already has (tags, a date range, a
camera, stars, picks) plus any words left over. Claude does the understanding when an API key
is set (it sees only the query, today's date, and the names of the tags, cameras, and
people in the library, never the photos, faces, or where they were taken); otherwise built-in
rules read dates, tags, and a few other phrases.

Left-over words are matched by what the photos look like when the vision model is
installed (its embeddings put words and photos side by side), and by file name otherwise.
"Best 30" asks for culling on top of the search (see `culling.py`).
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Sequence
from datetime import date
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

from photo_agent.advisor import Advisor, AdvisorError
from photo_agent.library import dates, faces, tagging
from photo_agent.library.core import Library, LibraryPhoto
from photo_agent.library.culling import CullRequest, cull_built_in
from photo_agent.library.filters import Flag, PhotoFilter
from photo_agent.library.vocabulary import TAGS, tags_in
from photo_agent.vision.worker import JobFailedError

log = logging.getLogger(__name__)

_FILLER = (
    "a an the of from with and in on at by for to my me our us we i show find get give all any "
    "some photo photos picture pictures pics pic image images shot shots taken during when "
    "where that this those these were was is are be please only just which who it its"
)
STOPWORDS = frozenset(_FILLER.split())
"""Words that say nothing about which photos are meant."""

MIN_VISUAL_SCORE = 0.03
"""How sure the vision model must be (0 to 1) that a photo shows the left-over words."""


class Understood(BaseModel):
    """What a search was taken to ask for."""

    tags: list[str] = Field(default_factory=list, description="Tags every photo must have.")
    people: list[str] = Field(default_factory=list, description="Named people who must be in it.")
    taken_after: date | None = None
    taken_before: date | None = None
    camera: str | None = None
    min_rating: int = Field(0, ge=0, le=5)
    flag: Flag | None = None
    best: int | None = Field(None, ge=1, le=500, description="Cull to this many best shots.")
    words: str = Field("", description="What is left, matched by look or by file name.")


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=300)
    album: str | None = Field(None, description="Search within this album.")


class SearchResult(BaseModel):
    query: str
    understood: Understood
    summary: str
    """The understanding in plain words, e.g. "Tagged beach and dog, taken Jun-Aug 2025"."""
    source: Literal["claude", "built-in"]
    matched_words_by: Literal["look", "file name"] | None = None
    """How the left-over words were matched, if there were any."""
    picked: str | None = Field(
        None, description="When the best few were asked for: how they were picked."
    )
    photos: list[LibraryPhoto]


def understand_built_in(
    query: str,
    today: date,
    cameras: list[str],
    people: Sequence[str] = (),
) -> Understood:
    text = f" {query.lower()} "
    out = Understood()
    best = re.search(r"\b(?:best|top)\s+(\d{1,3})\b", text)
    if best:
        out.best = max(1, min(500, int(best.group(1))))
        text = text.replace(best.group(0), " ")
    elif re.search(r"\b(?:the\s+)?best\b", text):
        out.best = 30
        text = re.sub(r"\b(?:the\s+)?best\b", " ", text)
    stars = re.search(r"\b([1-5])\s*(?:\+\s*)?stars?\b|\brated\s+([1-5])\b", text)
    if stars:
        out.min_rating = int(stars.group(1) or stars.group(2))
        text = text.replace(stars.group(0), " ")
    for words, flag in (
        (r"favou?rites?|picked|picks|keepers", "picked"),
        (r"rejected|rejects", "rejected"),
    ):
        if re.search(rf"\b(?:{words})\b", text):
            out.flag = flag  # type: ignore[assignment]
            text = re.sub(rf"\b(?:{words})\b", " ", text)
    span, text = dates.find_dates(text, today)
    if span:
        out.taken_after, out.taken_before = span
    for camera in sorted(cameras, key=len, reverse=True):
        if camera.lower() in text:
            out.camera = camera
            text = text.replace(camera.lower(), " ")
            break
    for person in sorted(people, key=len, reverse=True):
        pattern = rf"\b{re.escape(person.lower())}(?:'s)?\b"
        if re.search(pattern, text):
            out.people.append(person)
            text = re.sub(pattern, " ", text)
    out.tags, text = tags_in(text)
    out.words = " ".join(w for w in re.findall(r"[\w'&-]+", text) if w not in STOPWORDS)
    return out


SYSTEM = """You turn a photo library search into filters. Today is {today}.

Use only these tags (map everyday words onto them, e.g. "puppy" -> "dog"): {tags}.
Cameras in the library: {cameras}.
People named in the library (only these): {people}.

Dates: a span from taken_after to taken_before (inclusive, YYYY-MM-DD). Seasons are
meteorological in the northern hemisphere (summer is June 1 to August 31). "Last summer" is
the most recent summer that is over.
best: only when the person asks for the best, top, or nicest N photos (N, or 30 if no number).
flag: "picked" for favorites or picks; "rejected" for rejects. min_rating for stars.
words: anything else worth matching by how the photos look (e.g. "red umbrella"); leave out
filler such as "photos of" and anything already captured above. Leave fields empty rather
than guess."""


def _schema() -> dict[str, Any]:
    nullable_date = {"anyOf": [{"type": "string", "format": "date"}, {"type": "null"}]}
    return {
        "type": "object",
        "properties": {
            "tags": {"type": "array", "items": {"type": "string", "enum": list(TAGS)}},
            "people": {"type": "array", "items": {"type": "string"}},
            "taken_after": nullable_date,
            "taken_before": nullable_date,
            "camera": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "min_rating": {"type": "integer"},
            "flag": {
                "anyOf": [{"type": "string", "enum": ["picked", "rejected"]}, {"type": "null"}]
            },
            "best": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
            "words": {"type": "string"},
        },
        "required": [
            "tags",
            "people",
            "taken_after",
            "taken_before",
            "camera",
            "min_rating",
            "flag",
            "best",
            "words",
        ],
        "additionalProperties": False,
    }


async def understand_with_claude(
    advisor: Advisor, query: str, today: date, cameras: list[str], people: list[str]
) -> Understood:
    answer = await advisor.ask(
        system=SYSTEM.format(
            today=today.isoformat(),
            tags=", ".join(TAGS),
            cameras=", ".join(cameras) or "none",
            people=", ".join(people) or "none",
        ),
        content=[{"type": "text", "text": f"Search: {query}"}],
        schema=_schema(),
        tier="routine",
    )
    found = Understood.model_validate(
        {**answer, "min_rating": max(0, min(5, int(answer.get("min_rating") or 0)))}
    )
    # Keep to what the library knows; anything else is searched as words.
    unknown = [t for t in found.tags if t not in TAGS]
    found.tags = [t for t in found.tags if t in TAGS]
    found.people = [p for p in found.people if p in people]
    if found.camera not in cameras:
        found.camera = None
    found.words = " ".join([found.words, *unknown]).strip()
    return found


def describe(u: Understood) -> str:
    parts = []
    if u.tags:
        parts.append("tagged " + _join(u.tags))
    if u.people:
        parts.append("with " + _join(u.people))
    if u.taken_after and u.taken_before:
        if u.taken_after == u.taken_before:
            parts.append(f"taken {u.taken_after:%b %-d, %Y}")
        else:
            parts.append(f"taken {u.taken_after:%b %-d, %Y} to {u.taken_before:%b %-d, %Y}")
    elif u.taken_after:
        parts.append(f"taken since {u.taken_after:%b %-d, %Y}")
    elif u.taken_before:
        parts.append(f"taken before {u.taken_before:%b %-d, %Y}")
    if u.camera:
        parts.append(f"shot on {u.camera}")
    if u.min_rating:
        parts.append(f"{u.min_rating}+ stars")
    if u.flag:
        parts.append(u.flag)
    if u.words:
        parts.append(f"looking like “{u.words}”")
    text = ", ".join(parts) or "everything"
    if u.best:
        text = f"the best {u.best}: {text}"
    return text[0].upper() + text[1:]


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


async def search(
    library: Library,
    request: SearchRequest,
    advisor: Advisor | None,
    today: date | None = None,
) -> SearchResult:
    today = today or date.today()
    cameras = [
        r["camera"]
        for r in library.db.all("SELECT DISTINCT camera FROM photos WHERE camera IS NOT NULL")
    ]
    named = faces.names(library.db)
    people = list(named)
    source: Literal["claude", "built-in"] = "built-in"
    understood = None
    if advisor is not None:
        try:
            understood = await understand_with_claude(
                advisor, request.query, today, cameras, people
            )
            source = "claude"
        except (AdvisorError, ValueError, KeyError, TypeError) as exc:
            log.warning("Claude could not read the search, using built-in rules: %s", exc)
        except Exception as exc:  # Network trouble: built-in rules still give an answer.
            log.warning("Claude was unreachable for search: %s", exc)
    if understood is None:
        understood = understand_built_in(request.query, today, cameras, people)

    f = PhotoFilter(
        album=request.album,
        tags=understood.tags,
        people=[named[name] for name in understood.people],
        taken_after=understood.taken_after,
        taken_before=understood.taken_before,
        camera=understood.camera,
        min_rating=understood.min_rating,
        flag=understood.flag,
    )
    photos = library.photos(f)
    matched_by: Literal["look", "file name"] | None = None
    if understood.words:
        ranked = rank_by_look(library, photos, understood.words)
        if ranked is not None:
            photos, matched_by = ranked, "look"
        else:
            words = understood.words.lower().split()
            photos = [p for p in photos if all(w in p.filename.lower() for w in words)]
            matched_by = "file name"
    picked = None
    if understood.best:
        culled, _ = cull_built_in(
            library, CullRequest(count=understood.best, ids=[p.id for p in photos])
        )
        photos, picked = [pick.photo for pick in culled.picks], culled.summary
    return SearchResult(
        query=request.query,
        understood=understood,
        summary=describe(understood),
        source=source,
        matched_words_by=matched_by,
        picked=picked,
        photos=photos,
    )


def rank_by_look(
    library: Library, photos: list[LibraryPhoto], words: str
) -> list[LibraryPhoto] | None:
    """The photos that look like the words, best match first; None when no model can tell."""
    if library.worker is None:
        return None
    try:
        result = library.worker.run("embed_text", None, {"texts": [words]})
    except JobFailedError:
        return None
    text = np.asarray(result.value["vectors"][0], np.float32)
    scale, bias = float(result.value["scale"]), float(result.value["bias"])
    found = tagging.vectors(library.db, result.backend, [p.id for p in photos])
    scored = []
    for photo in photos:
        vector = found.get(photo.id)
        if vector is None:
            continue
        score = 1 / (1 + math.exp(-(float(vector @ text) * scale + bias)))
        if score >= MIN_VISUAL_SCORE:
            scored.append((score, photo))
    scored.sort(key=lambda t: -t[0])
    return [photo for _, photo in scored]
