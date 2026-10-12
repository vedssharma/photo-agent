"""People albums: faces found in library photos, grouped by who they look like.

Off until the person turns it on, and local: faces are found and compared on this machine
(by the model worker), their descriptions stay in the library database, and none of it is
sent to Claude. Only the names people give are used, to let search find "photos with
Maya". Turning it off forgets every face and person.

Grouping is incremental, so it keeps up as photos are indexed: each new face joins the
person whose faces it is closest to, if close enough; otherwise, if it matches another face
nobody has claimed, the two start a new person. Corrections stick: a face moved to someone
by hand stays there, and one taken away from a person is never put back automatically.
"""

from __future__ import annotations

import io
import json
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal

import numpy as np
import numpy.typing as npt
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from photo_agent import imaging
from photo_agent.library.db import LibraryDB

if TYPE_CHECKING:
    from photo_agent.library.core import Library

SETTING = "faces"

SAME_PERSON = {"yunet-sface": 0.42, "classical": 0.88}
"""How similar (cosine) two faces' descriptions must be to count as the same person, per
model. SFace's published threshold is 0.363; a little stricter keeps groups clean."""

Assigned = Literal["auto", "manual", "ignored"]


class PersonNotFoundError(KeyError):
    pass


class FaceNotFoundError(KeyError):
    pass


def enabled(db: LibraryDB) -> bool:
    row = db.one("SELECT value FROM settings WHERE key = ?", (SETTING,))
    return row is not None and row["value"] == "on"


def set_enabled(db: LibraryDB, on: bool) -> None:
    """Turn people albums on, or off (which forgets every face and person)."""
    with db.transaction():
        db.run(
            "INSERT INTO settings (key, value) VALUES (?, ?) "
            "ON CONFLICT (key) DO UPDATE SET value = excluded.value",
            (SETTING, "on" if on else "off"),
        )
        if not on:
            db.run("DELETE FROM faces")
            db.run("DELETE FROM people")
            db.run("DELETE FROM photo_stages WHERE stage = 'faces'")


def find_faces(library: Library, photo_id: str) -> None:
    """Indexing stage: find and describe the faces in one photo, then group them."""
    assert library.worker is not None
    result = library.worker.run("embed_faces", library.store.image(photo_id).proxy)
    model = result.backend
    db = library.db
    with db.transaction():
        db.run("DELETE FROM faces WHERE photo_id = ?", (photo_id,))
        new = []
        for face in result.value:
            face_id = uuid.uuid4().hex[:12]
            vector = np.asarray(face["embedding"], np.float32)
            db.run(
                "INSERT INTO faces (id, photo_id, box, model, embedding) VALUES (?, ?, ?, ?, ?)",
                (face_id, photo_id, json.dumps(face["box"]), model, vector.tobytes()),
            )
            new.append(face_id)
        for face_id in new:
            group(db, face_id)


def _vector(blob: bytes) -> npt.NDArray[np.float32]:
    return np.frombuffer(blob, np.float32)


def group(db: LibraryDB, face_id: str) -> str | None:
    """Put a face with the person it looks like, or start a person with a lookalike face
    nobody has claimed. Returns the person, if any."""
    face = db.one("SELECT * FROM faces WHERE id = ?", (face_id,))
    if face is None or face["assigned"] != "auto" or face["person_id"]:
        return face["person_id"] if face else None
    model, vector = face["model"], _vector(face["embedding"])
    threshold = SAME_PERSON.get(model, SAME_PERSON["classical"])
    others = db.all(
        "SELECT id, photo_id, person_id, embedding FROM faces "
        "WHERE model = ? AND id != ? AND assigned != 'ignored'",
        (model, face_id),
    )
    people: dict[str, list[npt.NDArray[np.float32]]] = {}
    loose: list[tuple[str, npt.NDArray[np.float32]]] = []
    for other in others:
        if other["person_id"]:
            people.setdefault(other["person_id"], []).append(_vector(other["embedding"]))
        elif other["photo_id"] != face["photo_id"]:  # one person appears once per photo
            loose.append((other["id"], _vector(other["embedding"])))
    best, best_score = None, threshold
    for person_id, members in people.items():
        centroid = np.mean(members, axis=0)
        centroid /= max(float(np.linalg.norm(centroid)), 1e-6)
        score = float(centroid @ vector)
        if score >= best_score:
            best, best_score = person_id, score
    if best is not None:
        db.run("UPDATE faces SET person_id = ? WHERE id = ?", (best, face_id))
        return best
    match = max(loose, key=lambda item: float(item[1] @ vector), default=None)
    if match is None or float(match[1] @ vector) < threshold:
        return None
    person_id = uuid.uuid4().hex[:12]
    db.run(
        "INSERT INTO people (id, name, created_at) VALUES (?, NULL, ?)",
        (person_id, datetime.now(UTC).isoformat()),
    )
    db.run("UPDATE faces SET person_id = ? WHERE id IN (?, ?)", (person_id, face_id, match[0]))
    return person_id


class Person(BaseModel):
    id: str
    name: str | None = Field(description="Given by the person using the app; None if unnamed.")
    photos: int = Field(description="How many photos they are in.")
    cover: str = Field(description="One of their faces, to show (see the face thumbnail).")


class Face(BaseModel):
    id: str
    box: list[float] = Field(description="Left, top, right, bottom, in fractions of the photo.")
    person: str | None
    person_name: str | None = None


class PersonChange(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    name: str | None = Field(None, max_length=60, description="A name, or null to clear it.")


class FaceChange(BaseModel):
    person: str | None = Field(
        description="The person this face is, or null for nobody (it is then left alone)."
    )


def people(db: LibraryDB) -> list[Person]:
    """Everyone found, named people first, then by how many photos they are in."""
    rows = db.all(
        "SELECT pe.id, pe.name, COUNT(DISTINCT f.photo_id) AS photos, MIN(f.id) AS cover "
        "FROM people pe JOIN faces f ON f.person_id = pe.id "
        "GROUP BY pe.id ORDER BY pe.name IS NULL, photos DESC, pe.name, pe.id"
    )
    return [Person(id=r["id"], name=r["name"], photos=r["photos"], cover=r["cover"]) for r in rows]


def names(db: LibraryDB) -> dict[str, str]:
    """Named people: name to id."""
    rows = db.all("SELECT id, name FROM people WHERE name IS NOT NULL ORDER BY name")
    return {r["name"]: r["id"] for r in rows}


def people_in(db: LibraryDB) -> dict[str, list[str]]:
    """Which people each photo has in it."""
    out: dict[str, list[str]] = {}
    for r in db.all(
        "SELECT DISTINCT photo_id, person_id FROM faces WHERE person_id IS NOT NULL "
        "ORDER BY photo_id, person_id"
    ):
        out.setdefault(r["photo_id"], []).append(r["person_id"])
    return out


def faces_in(db: LibraryDB, photo_id: str) -> list[Face]:
    rows = db.all(
        "SELECT f.id, f.box, f.person_id, pe.name FROM faces f "
        "LEFT JOIN people pe ON pe.id = f.person_id WHERE f.photo_id = ? ORDER BY f.box",
        (photo_id,),
    )
    return [
        Face(id=r["id"], box=json.loads(r["box"]), person=r["person_id"], person_name=r["name"])
        for r in rows
    ]


def rename(db: LibraryDB, person_id: str, change: PersonChange) -> Person:
    name = change.name or None
    if not db.run("UPDATE people SET name = ? WHERE id = ?", (name, person_id)):
        raise PersonNotFoundError(person_id)
    return _person(db, person_id)


def _person(db: LibraryDB, person_id: str) -> Person:
    for found in people(db):
        if found.id == person_id:
            return found
    raise PersonNotFoundError(person_id)


def merge(db: LibraryDB, person_id: str, into: str) -> Person:
    """The same person found twice: move every face of one to the other."""
    if person_id == into:
        return _person(db, into)
    for pid in (person_id, into):
        if db.one("SELECT 1 FROM people WHERE id = ?", (pid,)) is None:
            raise PersonNotFoundError(pid)
    with db.transaction():
        db.run("UPDATE faces SET person_id = ? WHERE person_id = ?", (into, person_id))
        db.run("DELETE FROM people WHERE id = ?", (person_id,))
    return _person(db, into)


def forget(db: LibraryDB, person_id: str) -> None:
    """Stop grouping these faces: the person goes, and their faces are left alone."""
    with db.transaction():
        db.run("UPDATE faces SET assigned = 'ignored' WHERE person_id = ?", (person_id,))
        if not db.run("DELETE FROM people WHERE id = ?", (person_id,)):
            raise PersonNotFoundError(person_id)


def move_face(db: LibraryDB, face_id: str, change: FaceChange) -> Face:
    face = db.one("SELECT photo_id FROM faces WHERE id = ?", (face_id,))
    if face is None:
        raise FaceNotFoundError(face_id)
    if change.person and db.one("SELECT 1 FROM people WHERE id = ?", (change.person,)) is None:
        raise PersonNotFoundError(change.person)
    assigned: Assigned = "manual" if change.person else "ignored"
    with db.transaction():
        db.run(
            "UPDATE faces SET person_id = ?, assigned = ? WHERE id = ?",
            (change.person, assigned, face_id),
        )
        # A person left with no faces is gone.
        db.run(
            "DELETE FROM people WHERE id NOT IN "
            "(SELECT person_id FROM faces WHERE person_id IS NOT NULL)"
        )
    return next(f for f in faces_in(db, face["photo_id"]) if f.id == face_id)


def face_thumbnail(library: Library, face_id: str, size: int = 160) -> bytes:
    """A square crop around one face, as JPEG."""
    row = library.db.one("SELECT photo_id, box FROM faces WHERE id = ?", (face_id,))
    if row is None:
        raise FaceNotFoundError(face_id)
    proxy = library.store.image(row["photo_id"]).proxy
    h, w = proxy.shape[:2]
    x0, y0, x1, y1 = json.loads(row["box"])
    side = max((x1 - x0) * w, (y1 - y0) * h) * 1.4
    cx, cy = (x0 + x1) / 2 * w, (y0 + y1) / 2 * h
    left, top = max(0, int(cx - side / 2)), max(0, int(cy - side / 2))
    crop = proxy[top : min(h, int(top + side)), left : min(w, int(left + side))]
    img = Image.fromarray(imaging.to_uint8(crop)).resize((size, size))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()
