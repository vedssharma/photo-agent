"""The library: importing many photos, the database, and background indexing."""

import io
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from photo_agent.library import Library
from photo_agent.library.db import MIGRATIONS, LibraryDB
from photo_agent.library.indexer import Indexer, Stage
from photo_agent.render import RenderCache
from photo_agent.settings import Settings
from photo_agent.store import DocumentStore

PHOTOS = ["landscape.png", "portrait.jpg", "low-light.jpg"]


def files(photos: Path, *names: str) -> list[tuple[str, tuple[str, bytes]]]:
    return [("files", (name, (photos / name).read_bytes())) for name in names]


def mean(image: Image.Image) -> float:
    return float(np.asarray(image.convert("L"), np.float32).mean())


def import_photos(client: TestClient, photos: Path, *names: str) -> dict[str, Any]:
    res = client.post("/api/library/photos", files=files(photos, *names))
    assert res.status_code == 201, res.text
    body: dict[str, Any] = res.json()
    return body


def test_imports_many_photos_at_once(client: TestClient, photos: Path) -> None:
    result = import_photos(client, photos, *PHOTOS)
    assert [p["filename"] for p in result["imported"]] == PHOTOS
    assert result["skipped"] == []

    listed = client.get("/api/library/photos").json()
    assert {p["filename"] for p in listed} == set(PHOTOS)
    assert all(not p["edited"] and p["revision"] == "original" for p in listed)
    # Each one is an ordinary document the editor can open.
    doc = client.get(f"/api/documents/{listed[0]['id']}").json()
    assert doc["filename"] == listed[0]["filename"]


def test_skips_copies_and_files_that_are_not_photos(client: TestClient, photos: Path) -> None:
    first = import_photos(client, photos, "portrait.jpg")["imported"][0]
    res = client.post(
        "/api/library/photos",
        files=[
            *files(photos, "portrait.jpg", "landscape.png"),
            ("files", ("notes.txt", b"not a photo")),
        ],
    )
    body = res.json()
    assert [p["filename"] for p in body["imported"]] == ["landscape.png"]
    reasons = {s["filename"]: s for s in body["skipped"]}
    assert reasons["portrait.jpg"]["existing"] == first["id"]
    assert "Already" in reasons["portrait.jpg"]["reason"]
    assert reasons["notes.txt"]["reason"]


def test_too_many_files_in_one_request(client: TestClient) -> None:
    many = [("files", (f"{i}.jpg", b"x")) for i in range(101)]
    assert client.post("/api/library/photos", files=many).status_code == 413


def test_photos_opened_in_the_editor_join_the_library(client: TestClient, upload: Any) -> None:
    doc = upload("portrait.jpg")
    photo = client.get(f"/api/library/photos/{doc['id']}").json()
    assert photo["filename"] == "portrait.jpg"


def test_thumbnails_follow_edits(client: TestClient, photos: Path) -> None:
    photo = import_photos(client, photos, "landscape.png")["imported"][0]
    res = client.get(f"/api/library/photos/{photo['id']}/thumbnail")
    assert res.status_code == 200
    original = Image.open(io.BytesIO(res.content))
    assert max(original.size) <= 480

    state = {
        "framing": [],
        "layers": [{"id": "l1", "name": "Darker", "operations": [{"op": "exposure", "stops": -2}]}],
    }
    client.post(f"/api/documents/{photo['id']}/edits", json={"label": "Darker", "state": state})
    edited = client.get(f"/api/library/photos/{photo['id']}").json()
    assert edited["edited"] and edited["revision"] != "original"

    darker = Image.open(
        io.BytesIO(
            client.get(
                f"/api/library/photos/{photo['id']}/thumbnail?revision={edited['revision']}"
            ).content
        )
    )
    assert mean(darker) < mean(original) * 0.6


def test_remove_deletes_the_photo_and_its_document(client: TestClient, photos: Path) -> None:
    photo = import_photos(client, photos, "portrait.jpg")["imported"][0]
    res = client.post("/api/library/photos/remove", json={"ids": [photo["id"], "000000000000"]})
    assert res.json() == {"removed": 1}
    assert client.get(f"/api/library/photos/{photo['id']}").status_code == 404
    assert client.get(f"/api/documents/{photo['id']}").status_code == 404
    # The same file can come back in.
    assert import_photos(client, photos, "portrait.jpg")["imported"]


def test_status_reports_indexing(client: TestClient, photos: Path) -> None:
    import_photos(client, photos, *PHOTOS)
    status = client.get("/api/library/status").json()
    assert status == {"total": 3, "pending": 0, "current": None}


def test_documents_made_before_the_library_join_it(settings: Settings, photos: Path) -> None:
    store = DocumentStore(settings.data_dir)
    doc = store.create("portrait.jpg", (photos / "portrait.jpg").read_bytes())
    library = Library(settings.data_dir, store, None, RenderCache(), "inline")
    assert [p.id for p in library.photos()] == [doc.id]
    assert library.indexer.status().pending == 0


def test_database_migrates_once(tmp_path: Path) -> None:
    db = LibraryDB(tmp_path / "library.db")
    assert db.version == len(MIGRATIONS)
    db.close()
    again = LibraryDB(tmp_path / "library.db")
    assert again.version == len(MIGRATIONS)


@pytest.fixture
def library(settings: Settings, photos: Path) -> Library:
    store = DocumentStore(settings.data_dir)
    library = Library(settings.data_dir, store, None, RenderCache(), "inline")
    for name in PHOTOS:
        library.import_file(name, (photos / name).read_bytes())
    return library


def test_stages_run_once_per_version_and_failures_do_not_block(library: Library) -> None:
    runs: list[tuple[str, str]] = []

    def ok(lib: Library, photo_id: str) -> None:
        runs.append(("ok", photo_id))

    def broken(lib: Library, photo_id: str) -> None:
        runs.append(("broken", photo_id))
        raise ValueError("odd file")

    indexer = Indexer(
        library, [Stage("ok", 1, "OK", ok), Stage("broken", 1, "Broken", broken)], "inline"
    )
    indexer.kick()
    assert len(runs) == 6
    assert indexer.status().pending == 0
    errors = library.db.all("SELECT error FROM photo_stages WHERE stage = 'broken'")
    assert [r["error"] for r in errors] == ["odd file"] * 3

    indexer.kick()
    assert len(runs) == 6  # nothing new to do

    # A new version of one stage re-runs only that stage.
    indexer.stages = [Stage("ok", 2, "OK", ok), Stage("broken", 1, "Broken", broken)]
    indexer.kick()
    assert [kind for kind, _ in runs[6:]] == ["ok"] * 3


def test_disabled_stages_wait_until_enabled(library: Library) -> None:
    on = False
    runs: list[str] = []
    stage = Stage("opt_in", 1, "Opt in", lambda lib, pid: runs.append(pid), lambda lib: on)
    indexer = Indexer(library, [stage], "inline")
    indexer.kick()
    assert runs == []
    on = True
    assert indexer.status().pending == 3
    indexer.kick()
    assert len(runs) == 3


def test_background_indexing_runs_on_a_thread(library: Library) -> None:
    runs: list[str] = []
    indexer = Indexer(
        library, [Stage("slow", 1, "Slow", lambda lib, pid: runs.append(pid))], "background"
    )
    indexer.kick()
    deadline = time.monotonic() + 10
    while indexer.status().pending and time.monotonic() < deadline:
        time.sleep(0.02)
    indexer.stop()
    assert len(runs) == 3
