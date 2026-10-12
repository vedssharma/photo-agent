"""Culling: picking the best shots from a shoot, the way a photographer would.

Photos are first gathered into moments: shots of the same thing taken together (a burst,
the same pose a few times over), recognized by being taken within seconds of each other
and looking alike, or by looking nearly the same whenever they were taken. The best shot of
each moment is picked, so the picks cover the whole shoot instead of ten frames of the same
sunset; only when there are fewer moments than picks wanted does a moment get a second.

"Best" weighs what indexing measured (see `vision/quality.py`): sharpness most, then
exposure, eyes open, and appeal, plus what the person has already said (stars, picks).
Rejected photos are never picked. Every pick comes with its reasons; Claude, when an API
key is set, rewrites them and the summary in plain words from the measurements (it sees
numbers and file names, not the photos).
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

from photo_agent.advisor import Advisor, AdvisorError
from photo_agent.library import fingerprints, stages
from photo_agent.library.core import Library, LibraryPhoto
from photo_agent.library.db import placeholders
from photo_agent.library.filters import PhotoFilter

log = logging.getLogger(__name__)

SAME_SHOT = 12
"""Fingerprints this close (bits of 64) are the same shot, whenever taken."""
SAME_MOMENT = 24
"""Fingerprints this close, taken within `BURST_SECONDS`, are the same moment."""
BURST_SECONDS = 30
TOO_SIMILAR = 6
"""A second pick from a moment must differ from the first by more than this."""

WEIGHTS = {"sharpness": 0.4, "exposure": 0.25, "eyes": 0.15, "appeal": 0.2}


class CullRequest(BaseModel):
    count: int = Field(30, ge=1, le=500, description="How many photos to pick.")
    ids: list[str] | None = Field(
        None, max_length=10000, description="Pick from these photos (else from the filter)."
    )
    filter: PhotoFilter | None = Field(None, description="Pick from the photos it lets through.")
    explain: bool = Field(True, description="Ask Claude to explain the picks, when it can.")


class Pick(BaseModel):
    photo: LibraryPhoto
    score: float = Field(description="0 to 1: how good a shot it is, all things weighed.")
    moment: int = Field(description="Which moment it is from (numbered in time order).")
    moment_size: int = Field(description="How many shots that moment has.")
    reasons: list[str] = Field(description="Why it was picked, in plain words.")


class CullResult(BaseModel):
    picks: list[Pick] = Field(description="Best first.")
    considered: int
    moments: int
    summary: str
    source: Literal["claude", "built-in"]


class _Shot:
    """A photo with what culling needs to know about it."""

    def __init__(self, photo: LibraryPhoto, row: Any) -> None:
        self.photo = photo
        self.sharpness = float(row["sharpness"] or 0)
        self.exposure = float(row["exposure"] or 0)
        self.appeal = float(row["appeal"] or 0)
        self.faces = int(row["faces"] or 0)
        self.eyes_closed = int(row["eyes_closed"] or 0)
        self.fingerprint = int(row["fingerprint"], 16) if row["fingerprint"] else None
        # Camera time has no zone; compare it with import times on the same footing.
        self.when: datetime = (photo.taken_at or photo.imported_at).replace(tzinfo=None)
        self.timed = photo.taken_at is not None
        eyes = 1.0 if self.faces == 0 else 1 - self.eyes_closed / self.faces
        self.eyes = eyes
        measured = (
            WEIGHTS["sharpness"] * self.sharpness
            + WEIGHTS["exposure"] * self.exposure
            + WEIGHTS["eyes"] * eyes
            + WEIGHTS["appeal"] * self.appeal
        )
        # What the person already said (a pick, stars) counts for a quarter.
        said = 1.0 if photo.flag == "picked" else photo.rating / 5
        self.score = 0.75 * measured + 0.25 * said

    def apart(self, other: _Shot) -> int:
        if self.fingerprint is None or other.fingerprint is None:
            return 64
        return (self.fingerprint ^ other.fingerprint).bit_count()


def _shots(library: Library, request: CullRequest) -> list[_Shot]:
    photos = library.photos(request.filter or PhotoFilter())
    if request.ids is not None:
        wanted = set(request.ids)
        photos = [p for p in photos if p.id in wanted]
    photos = [p for p in photos if p.flag != "rejected"]
    ids = [p.id for p in photos]
    if not ids:
        return []

    def rows() -> dict[str, Any]:
        return {
            r["id"]: r
            for r in library.db.all(
                "SELECT id, sharpness, exposure, appeal, faces, eyes_closed, fingerprint "
                f"FROM photos WHERE id IN ({placeholders(ids)})",
                ids,
            )
        }

    found = rows()
    # Photos indexing has not reached yet are measured now.
    missing = [i for i in ids if found[i]["sharpness"] is None or found[i]["fingerprint"] is None]
    for photo_id in missing:
        stages.assess(library, photo_id)
        fingerprints.fingerprint_photo(library, photo_id)
    if missing:
        found = rows()
    return [_Shot(p, found[p.id]) for p in photos]


def moments(shots: list[_Shot]) -> list[list[_Shot]]:
    """Shots grouped into moments, in time order; each moment's shots best first."""
    ordered = sorted(shots, key=lambda s: (s.when, s.photo.id))
    parent = list(range(len(ordered)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    # Each shot against every later one, a row at a time (vectorized: shoots run to
    # thousands of photos).
    prints = np.array([s.fingerprint or 0 for s in ordered], np.uint64)
    has_print = np.array([s.fingerprint is not None for s in ordered])
    seconds = np.array([s.when.timestamp() for s in ordered])
    timed = np.array([s.timed for s in ordered])
    for i in range(len(ordered) - 1):
        distance = np.bitwise_count(prints[i + 1 :] ^ prints[i])
        together = timed[i] & timed[i + 1 :] & (seconds[i + 1 :] - seconds[i] <= BURST_SECONDS)
        linked = has_print[i] & has_print[i + 1 :]
        linked &= (distance <= SAME_SHOT) | (together & (distance <= SAME_MOMENT))
        for j in np.flatnonzero(linked):
            parent[root(i + 1 + int(j))] = root(i)
    groups: dict[int, list[_Shot]] = {}
    for i, shot in enumerate(ordered):
        groups.setdefault(root(i), []).append(shot)
    return [sorted(g, key=lambda s: -s.score) for g in groups.values()]


def choose(groups: list[list[_Shot]], count: int) -> list[tuple[_Shot, int]]:
    """The best shot of each moment, best moments first; then seconds, if there is room.
    Returns each pick with the index of its moment."""
    picks: list[tuple[_Shot, int]] = []
    taken: dict[int, list[_Shot]] = {}
    while len(picks) < count:
        best_left = []
        for index, group in enumerate(groups):
            chosen = taken.get(index, [])
            for shot in group:
                if shot in chosen or any(shot.apart(c) <= TOO_SIMILAR for c in chosen):
                    continue
                best_left.append((shot, index))
                break
        if not best_left:
            break
        best_left.sort(key=lambda item: -item[0].score)
        for shot, index in best_left[: count - len(picks)]:
            picks.append((shot, index))
            taken.setdefault(index, []).append(shot)
    picks.sort(key=lambda item: -item[0].score)
    return picks


def reasons(shot: _Shot, group: list[_Shot]) -> list[str]:
    """Why a shot was picked, in plain words."""
    out = []
    others = [s for s in group if s is not shot]
    if others:
        out.append(f"Best of {len(group)} similar shots")
        if shot.sharpness >= max(s.sharpness for s in others) + 0.03:
            out.append("sharpest of them")
        if shot.faces and not shot.eyes_closed and any(s.eyes_closed for s in others):
            out.append("eyes open where others blinked")
        if shot.exposure >= max(s.exposure for s in others) + 0.05:
            out.append("best exposed")
    else:
        if shot.sharpness >= 0.75:
            out.append("Sharp")
        if shot.exposure >= 0.9:
            out.append("well exposed")
        if shot.appeal >= 0.75:
            out.append("strong color and contrast")
        if shot.faces and not shot.eyes_closed:
            out.append("eyes open")
    if shot.photo.flag == "picked":
        out.append("you picked it")
    elif shot.photo.rating:
        out.append(f"you gave it {shot.photo.rating} star{'s' if shot.photo.rating > 1 else ''}")
    if not out:
        out.append("The best of what was left")
    out[0] = out[0][0].upper() + out[0][1:]
    return out


def _summary(considered: int, groups: list[list[_Shot]], picks: list[tuple[_Shot, int]]) -> str:
    shots = [s for g in groups for s in g]
    soft = sum(1 for s in shots if s.sharpness < 0.35)
    blinks = sum(1 for s in shots if s.eyes_closed)
    text = (
        f"Picked {len(picks)} of {considered} photo{'s' if considered != 1 else ''}"
        f" from {len(groups)} moment{'s' if len(groups) != 1 else ''}"
    )
    if len(picks) < len(groups):
        text += ": the best shot of the strongest moments"
    elif len(picks) == len(groups):
        text += ": the best shot of each"
    else:
        text += ": the best of each, and a second from some"
    passed = []
    if soft:
        passed.append(f"{soft} soft or blurry")
    if blinks:
        passed.append(f"{blinks} with closed eyes")
    if passed:
        text += ". Among the rest: " + " and ".join(passed)
    return text + "."


SYSTEM = """You help someone choose the best photos from a shoot. The picks were made by
measurements: sharpness, exposure, eyes open, and a rough color-and-contrast appeal score,
all from 0 to 1, with shots grouped into moments (several frames of the same thing). Write
a two-sentence summary of what was picked and why, and for each pick one short, friendly
reason (under 15 words) drawn from its numbers and notes. Do not invent anything the
numbers do not show; you cannot see the photos."""


def _schema(count: int) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "reasons": {
                "type": "array",
                "items": {"type": "string"},
                "description": f"One per pick, in order ({count}).",
            },
        },
        "required": ["summary", "reasons"],
        "additionalProperties": False,
    }


async def explain_with_claude(
    advisor: Advisor, result: CullResult, shots: dict[str, _Shot]
) -> CullResult:
    lines = [result.summary, ""]
    for i, pick in enumerate(result.picks, 1):
        s = shots[pick.photo.id]
        lines.append(
            f"{i}. {pick.photo.filename}: sharpness {s.sharpness:.2f}, exposure "
            f"{s.exposure:.2f}, appeal {s.appeal:.2f}, faces {s.faces}, closed eyes "
            f"{s.eyes_closed}, moment of {pick.moment_size}; notes: {'; '.join(pick.reasons)}"
        )
    answer = await advisor.ask(
        system=SYSTEM,
        content=[{"type": "text", "text": "\n".join(lines)}],
        schema=_schema(len(result.picks)),
        tier="routine",
    )
    written = [str(r).strip() for r in answer["reasons"]]
    if len(written) != len(result.picks) or not str(answer["summary"]).strip():
        raise ValueError("Claude's explanation did not match the picks")
    picks = [
        pick.model_copy(update={"reasons": [reason]})
        for pick, reason in zip(result.picks, written, strict=True)
    ]
    return result.model_copy(
        update={"picks": picks, "summary": str(answer["summary"]).strip(), "source": "claude"}
    )


def cull_built_in(library: Library, request: CullRequest) -> tuple[CullResult, dict[str, _Shot]]:
    shots = _shots(library, request)
    groups = moments(shots)
    chosen = choose(groups, request.count)
    picks = [
        Pick(
            photo=shot.photo,
            score=round(float(np.clip(shot.score, 0, 1)), 3),
            moment=index + 1,
            moment_size=len(groups[index]),
            reasons=reasons(shot, groups[index]),
        )
        for shot, index in chosen
    ]
    result = CullResult(
        picks=picks,
        considered=len(shots),
        moments=len(groups),
        summary=_summary(len(shots), groups, chosen) if shots else "No photos to pick from.",
        source="built-in",
    )
    return result, {s.photo.id: s for s in shots}


async def cull(library: Library, request: CullRequest, advisor: Advisor | None) -> CullResult:
    result, shots = cull_built_in(library, request)
    if advisor is None or not request.explain or not result.picks:
        return result
    try:
        return await explain_with_claude(advisor, result, shots)
    except (AdvisorError, ValueError, KeyError, TypeError) as exc:
        log.warning("Claude could not explain the picks: %s", exc)
    except Exception as exc:  # Network trouble: the built-in reasons still stand.
        log.warning("Claude was unreachable to explain the picks: %s", exc)
    return result
