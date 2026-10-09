import io
from collections.abc import Callable
from typing import Any

import numpy as np
from fastapi.testclient import TestClient
from helpers import step
from PIL import Image

from photo_agent.graph import DocumentView
from photo_agent.operations import Crop, Exposure
from photo_agent.routes import get_store
from photo_agent.settings import Settings

Upload = Callable[[str], dict[str, Any]]


def test_upload_heic_creates_a_document(upload: Upload, settings: Settings) -> None:
    doc = upload("phone.heic")
    assert doc["filename"] == "phone.heic"
    assert doc["format"] == "HEIF"
    assert (doc["width"], doc["height"]) == (451, 300)
    assert doc["state"] == {"framing": [], "layers": [], "cutout": None}
    assert doc["revision"] == "original"
    assert (settings.data_dir / "documents" / doc["id"] / "original.heic").is_file()


def test_get_document(client: TestClient, upload: Upload) -> None:
    doc = upload("portrait.jpg")
    res = client.get(f"/api/documents/{doc['id']}")
    assert res.status_code == 200
    assert res.json() == doc


def test_unknown_document_is_404(client: TestClient) -> None:
    assert client.get("/api/documents/000000000000").status_code == 404
    assert client.get("/api/documents/..%2F..%2Fetc").status_code == 404


def test_rejects_files_that_are_not_photos(client: TestClient) -> None:
    res = client.post("/api/documents", files={"file": ("notes.txt", b"hello")})
    assert res.status_code == 415
    assert "image" in res.json()["detail"]


def test_upload_iphone_mpo_jpeg(client: TestClient, settings: Settings) -> None:
    buf = io.BytesIO()
    Image.new("RGB", (40, 30), (90, 120, 200)).save(
        buf, format="MPO", save_all=True, append_images=[Image.new("L", (20, 15))]
    )
    res = client.post("/api/documents", files={"file": ("IMG_8684.jpeg", buf.getvalue())})
    assert res.status_code == 201, res.text
    doc = res.json()
    assert doc["format"] == "JPEG"
    assert (settings.data_dir / "documents" / doc["id"] / "original.jpg").is_file()
    assert client.get(f"/api/documents/{doc['id']}/original").status_code == 200


def test_upload_keeps_only_the_base_filename(client: TestClient, photos: Any) -> None:
    data = (photos / "portrait.jpg").read_bytes()
    res = client.post("/api/documents", files={"file": ("../../evil/me.jpg", data)})
    assert res.json()["filename"] == "me.jpg"


def test_original_is_served_upright_as_jpeg(client: TestClient, upload: Upload) -> None:
    doc = upload("phone.heic")
    res = client.get(f"/api/documents/{doc['id']}/original")
    assert res.status_code == 200
    assert res.headers["content-type"] == "image/jpeg"
    img = Image.open(io.BytesIO(res.content))
    assert img.size == (451, 300)


def test_undo_and_redo(client: TestClient, upload: Upload, settings: Settings) -> None:
    doc = upload("portrait.jpg")
    store = get_store(settings)
    stored = store.get(doc["id"])
    stored.commit(step("brighter", Exposure(stops=1)))
    store.save(stored)

    res = client.post(f"/api/documents/{doc['id']}/undo")
    assert res.json()["state"]["layers"] == []
    assert res.json()["can_redo"] is True

    res = client.post(f"/api/documents/{doc['id']}/redo")
    assert res.json()["state"]["layers"][0]["operations"][0]["stops"] == 1
    assert res.json()["can_undo"] is True


def test_preview_renders_current_operations(
    client: TestClient, upload: Upload, settings: Settings
) -> None:
    doc = upload("portrait.jpg")
    store = get_store(settings)
    stored = store.get(doc["id"])
    stored.commit(step("square", Exposure(stops=1)))
    store.save(stored)

    res = client.get(f"/api/documents/{doc['id']}/preview")
    assert res.status_code == 200
    assert res.headers["content-type"] == "image/jpeg"
    edited = Image.open(io.BytesIO(res.content)).convert("L")
    original = Image.open(
        io.BytesIO(client.get(f"/api/documents/{doc['id']}/original").content)
    ).convert("L")
    assert np.asarray(edited).mean() > np.asarray(original).mean()


def test_before_view_shares_the_current_framing(
    client: TestClient, upload: Upload, settings: Settings
) -> None:
    doc = upload("landscape.png")
    store = get_store(settings)
    stored = store.get(doc["id"])
    stored.commit(step("square", Exposure(stops=1), Crop(aspect="1:1")))
    store.save(stored)

    before = Image.open(io.BytesIO(client.get(f"/api/documents/{doc['id']}/before").content))
    after = Image.open(io.BytesIO(client.get(f"/api/documents/{doc['id']}/preview").content))
    assert before.size == after.size == (427, 427)
    assert np.asarray(before.convert("L")).mean() < np.asarray(after.convert("L")).mean()


def test_checkout_jumps_to_any_step(client: TestClient, upload: Upload, settings: Settings) -> None:
    doc = upload("portrait.jpg")
    store = get_store(settings)
    stored = store.get(doc["id"])
    stored.commit(step("brighter", Exposure(stops=1)))
    stored.commit(step("much brighter", Exposure(stops=2)))
    store.save(stored)
    first = stored.steps[0].id

    res = client.post(f"/api/documents/{doc['id']}/checkout", json={"step_id": first})
    assert res.status_code == 200
    body = res.json()
    assert body["head"] == first
    assert [s["active"] for s in body["history"]] == [True, False]
    assert body["redo_label"] == "much brighter"

    res = client.post(f"/api/documents/{doc['id']}/checkout", json={"step_id": None})
    assert res.json()["revision"] == "original"
    res = client.post(f"/api/documents/{doc['id']}/checkout", json={"step_id": "nope"})
    assert res.status_code == 404


def test_manual_edits_become_history_steps(
    client: TestClient, upload: Upload, settings: Settings
) -> None:
    doc = upload("portrait.jpg")
    store = get_store(settings)
    stored = store.get(doc["id"])
    stored.commit(step("brighter", Exposure(stops=1)))
    store.save(stored)
    state = DocumentView.of(stored).model_dump(mode="json")["state"]

    state["layers"][0]["opacity"] = 40
    edit = {"label": "Brighter opacity 40%", "state": state, "coalesce": "opacity"}
    res = client.post(f"/api/documents/{doc['id']}/edits", json=edit)
    assert res.status_code == 200
    body = res.json()
    assert body["state"]["layers"][0]["opacity"] == 40
    assert [(s["kind"], s["label"]) for s in body["history"]] == [
        ("agent", "brighter"),
        ("manual", "Brighter opacity 40%"),
    ]
    assert body["chat"][-1]["text"] == "Edited by hand: Brighter opacity 40%"

    state["layers"][0]["opacity"] = 30
    edit = {"label": "Brighter opacity 30%", "state": state, "coalesce": "opacity"}
    body = client.post(f"/api/documents/{doc['id']}/edits", json=edit).json()
    assert len(body["history"]) == 2

    state["layers"][0]["operations"] = [{"op": "crop", "aspect": "1:1"}]
    res = client.post(f"/api/documents/{doc['id']}/edits", json={"label": "x", "state": state})
    assert res.status_code == 422
