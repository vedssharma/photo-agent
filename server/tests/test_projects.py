import io
import json
import zipfile
from collections.abc import Callable
from typing import Any

from fastapi.testclient import TestClient
from helpers import step

from photo_agent import projects
from photo_agent.main import app
from photo_agent.operations import Exposure
from photo_agent.routes import get_store
from photo_agent.settings import Settings, get_settings

Upload = Callable[[str], dict[str, Any]]


def edited(upload: Upload, settings: Settings) -> dict[str, Any]:
    doc = upload("phone.heic")
    store = get_store(settings)
    stored = store.get(doc["id"])
    stored.commit(step("brighter", Exposure(stops=0.5)))
    store.save(stored)
    return doc


def test_project_round_trip(client: TestClient, upload: Upload, settings: Settings) -> None:
    doc = edited(upload, settings)
    res = client.get(f"/api/documents/{doc['id']}/project")
    assert res.status_code == 200
    assert res.headers["content-type"] == projects.MEDIA_TYPE
    assert "phone.photoagent" in res.headers["content-disposition"]
    with zipfile.ZipFile(io.BytesIO(res.content)) as zf:
        assert sorted(zf.namelist()) == ["document.json", "manifest.json", "original.heic"]

    reopened = client.post("/api/projects", files={"file": ("phone.photoagent", res.content)})
    assert reopened.status_code == 201, reopened.text
    body = reopened.json()
    # The original document still exists here, so the reopened copy gets a new id.
    assert body["id"] != doc["id"]
    assert body["filename"] == "phone.heic"
    assert [s["label"] for s in body["history"]] == ["brighter"]
    assert body["revision"] != "original"
    assert client.get(f"/api/documents/{body['id']}/preview").status_code == 200


def test_restore_from_browser_autosave(
    client: TestClient, upload: Upload, settings: Settings, tmp_path: Any
) -> None:
    doc = edited(upload, settings)
    source = client.get(f"/api/documents/{doc['id']}/source")
    assert source.headers["content-type"] == "image/heic"
    graph = client.get(f"/api/documents/{doc['id']}/graph").json()
    assert graph["steps"][0]["label"] == "brighter"

    # Simulate a server whose data folder was wiped.
    fresh = settings.model_copy(update={"data_dir": tmp_path / "fresh"})
    app.dependency_overrides[get_settings] = lambda: fresh
    res = client.post(
        "/api/projects",
        files={"original": ("phone.heic", source.content)},
        data={"graph": json.dumps(graph)},
    )
    assert res.status_code == 201, res.text
    assert res.json()["id"] == doc["id"]
    assert res.json()["head"] == graph["head"]


def test_rejects_bad_projects(client: TestClient, photos: Any) -> None:
    res = client.post("/api/projects", files={"file": ("x.photoagent", b"not a zip")})
    assert res.status_code == 415
    assert "not a photo-agent project" in res.json()["detail"]

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("manifest.json", json.dumps({"format": "photo-agent-project", "version": 1}))
        zf.writestr("document.json", json.dumps({"filename": "a.jpg"}))
        zf.writestr("original.jpg", (photos / "portrait.jpg").read_bytes())
    res = client.post("/api/projects", files={"file": ("x.photoagent", buf.getvalue())})
    assert res.status_code == 415

    assert client.post("/api/projects").status_code == 415


def test_rejects_edits_for_a_different_photo(
    client: TestClient, upload: Upload, settings: Settings, photos: Any
) -> None:
    doc = edited(upload, settings)
    graph = client.get(f"/api/documents/{doc['id']}/graph").text
    res = client.post(
        "/api/projects",
        files={"original": ("p.jpg", (photos / "portrait.jpg").read_bytes())},
        data={"graph": graph},
    )
    assert res.status_code == 415
    assert "does not match" in res.json()["detail"]
