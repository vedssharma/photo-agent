"""Batch editing many photos at once, and exporting them together."""

import io
import zipfile
from pathlib import Path
from typing import Any

import pytest
from fake_model import FakeModel, text, tool
from fastapi.testclient import TestClient
from library_helpers import PHOTOS, import_bytes, import_photos, jpeg_with_exif

from photo_agent.main import app
from photo_agent.routes import get_model


@pytest.fixture
def ids(client: TestClient, photos: Path) -> list[str]:
    """The three fixture photos, in the order imported."""
    imported = import_photos(client, photos, *PHOTOS)["imported"]
    return [p["id"] for p in imported]


def state(client: TestClient, doc_id: str) -> dict[str, Any]:
    found: dict[str, Any] = client.get(f"/api/documents/{doc_id}").json()["state"]
    return found


def edit(client: TestClient, doc_id: str, *layers: dict[str, Any]) -> None:
    body = {"label": "test", "state": {**state(client, doc_id), "layers": list(layers)}}
    res = client.post(f"/api/documents/{doc_id}/edits", json=body)
    assert res.status_code == 200, res.text


def layer(name: str, *ops: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {"id": name.lower().replace(" ", "-"), "name": name, "operations": list(ops), **extra}


def ops_of(client: TestClient, doc_id: str) -> list[tuple[str, list[str]]]:
    return [
        (layer["name"], [op["op"] for op in layer["operations"]])
        for layer in state(client, doc_id)["layers"]
    ]


def run_batch(client: TestClient, **body: Any) -> dict[str, Any]:
    res = client.post("/api/library/batch", json=body)
    assert res.status_code == 202, res.text
    # Tests run the batch before the response returns; the status shows how it went.
    found = client.get(f"/api/library/batch/{res.json()['id']}")
    assert found.status_code == 200
    status: dict[str, Any] = found.json()
    assert status["state"] == "done", status
    return status


WARM = {"op": "white_balance", "temperature": 25}
BRIGHT = {"op": "exposure", "stops": 0.4}


def test_copies_one_photos_look_to_the_others(client: TestClient, ids: list[str]) -> None:
    a, b, c = ids
    edit(client, a, layer("Warm and bright", WARM, BRIGHT))
    edit(client, b, layer("Moody", {"op": "contrast", "amount": 40}))
    status = run_batch(client, ids=ids, reference=a, match=0)
    assert (status["total"], status["done"], status["failed"]) == (3, 3, [])
    assert ops_of(client, a) == [("Warm and bright", ["white_balance", "exposure"])]
    # B's own look gives way to the shared one.
    assert ops_of(client, b) == [("Warm and bright", ["white_balance", "exposure"])]
    assert ops_of(client, c) == [("Warm and bright", ["white_balance", "exposure"])]
    # Each photo's layers are its own.
    assert state(client, b)["layers"][0]["id"] != state(client, c)["layers"][0]["id"]
    # One step in each history: undo brings B's old look back.
    client.post(f"/api/documents/{b}/undo")
    assert ops_of(client, b) == [("Moody", ["contrast"])]


def test_evens_out_each_photo_toward_the_reference(client: TestClient, ids: list[str]) -> None:
    a, b, c = ids
    edit(client, a, layer("Warm", WARM))
    run_batch(client, ids=ids, reference=a, match=60)
    assert ops_of(client, a) == [("Warm", ["white_balance"])]
    for other in (b, c):
        warm, matched = state(client, other)["layers"]
        assert warm["name"] == "Warm"
        assert matched["name"] == "Match the set"
        (op,) = matched["operations"]
        assert op["op"] == "match_reference"
        assert op["amount"] == 60
        assert op["stats"] is not None
    # Another batch replaces the evening-out rather than stacking it.
    run_batch(client, ids=ids, reference=a, match=30)
    assert [lyr["name"] for lyr in state(client, b)["layers"]] == ["Warm", "Match the set"]
    # The thumbnails show the new look.
    assert client.get(f"/api/library/photos/{b}").json()["edited"] is True


def test_keeps_each_photos_local_work(client: TestClient, ids: list[str]) -> None:
    a, b, _ = ids
    edit(client, a, layer("Warm", WARM))
    brushed = layer(
        "Brighten her face",
        BRIGHT,
        mask={"kind": "brush", "strokes": [{"points": [[0.4, 0.4], [0.5, 0.5]], "size": 0.05}]},
    )
    edit(client, b, layer("Old look", {"op": "saturation", "amount": -30}), brushed)
    run_batch(client, ids=[a, b], reference=a, match=0)
    assert ops_of(client, b) == [("Brighten her face", ["exposure"]), ("Warm", ["white_balance"])]


def test_applies_a_recipe_to_every_photo(client: TestClient, ids: list[str]) -> None:
    res = client.post(
        "/api/recipes", json={"name": "Golden", "layers": [layer("Golden", WARM, BRIGHT)]}
    )
    recipe = res.json()["id"]
    run_batch(client, ids=ids, recipe=recipe)
    for doc_id in ids:
        assert ops_of(client, doc_id)[0] == ("Golden", ["white_balance", "exposure"])
    assert ops_of(client, ids[0]) == [("Golden", ["white_balance", "exposure"])]
    assert [name for name, _ in ops_of(client, ids[1])] == ["Golden", "Match the set"]


def test_the_agent_works_out_the_look_from_an_instruction(
    client: TestClient, ids: list[str]
) -> None:
    model = FakeModel(
        [tool("white_balance", {"temperature": 20}), tool("exposure", {"stops": 0.3})],
        [text("Warmed and brightened it; the others follow.")],
    )
    app.dependency_overrides[get_model] = lambda: model
    status = run_batch(client, ids=ids, instruction="make these warm and bright", match=50)
    assert status["message"] == "Warmed and brightened it; the others follow."
    assert status["reference"] == ids[0]
    assert [op for _, ops in ops_of(client, ids[0]) for op in ops] == [
        "white_balance",
        "exposure",
    ]
    for other in ids[1:]:
        found = ops_of(client, other)
        assert found[0][1] == ["white_balance", "exposure"]
        assert found[-1][0] == "Match the set"
    # The reference's chat records the turn, like any other.
    doc = client.get(f"/api/documents/{ids[0]}").json()
    assert doc["chat"][0]["text"] == "make these warm and bright"


def test_refuses_what_it_cannot_do(client: TestClient, ids: list[str]) -> None:
    res = client.post("/api/library/batch", json={"ids": ids, "instruction": "warmer"})
    assert res.status_code == 400
    assert "API key" in res.json()["detail"]
    res = client.post("/api/library/batch", json={"ids": ids, "recipe": "nope"})
    assert res.status_code == 404
    res = client.post("/api/library/batch", json={"ids": ["nope"]})
    assert res.status_code == 404
    res = client.post("/api/library/batch", json={"ids": ids[:1], "reference": ids[2]})
    assert res.status_code == 422
    assert client.get("/api/library/batch/nope").status_code == 404


def test_exports_photos_together(client: TestClient, ids: list[str]) -> None:
    edit(client, ids[0], layer("Warm", WARM))
    # Another photo with the same name.
    (again,) = import_bytes(client, ("portrait.jpg", jpeg_with_exif()))
    res = client.post(
        "/api/library/export",
        json={"ids": [*ids, again["id"]], "options": {"format": "jpeg", "quality": 80}},
    )
    assert res.status_code == 200, res.text
    assert res.headers["content-type"] == "application/zip"
    with zipfile.ZipFile(io.BytesIO(res.content)) as archive:
        names = archive.namelist()
        assert sorted(names) == [
            "landscape-edited.jpg",
            "low-light-edited.jpg",
            "portrait-edited (2).jpg",
            "portrait-edited.jpg",
        ]
        assert archive.read("landscape-edited.jpg")[:2] == b"\xff\xd8"


def test_from_a_trip_to_the_best_shots_with_one_look(client: TestClient, photos: Path) -> None:
    """The roadmap's whole flow, small: import a shoot, ask for the best, give them one
    look with one instruction, and download them."""
    from PIL import Image, ImageFilter

    view = Image.open(photos / "landscape.png")
    night = Image.open(photos / "low-light.jpg")
    shots = [
        (f"IMG_{i}.jpg", image, f"2026:07:04 {time}")
        for i, (image, time) in enumerate(
            [
                (view.filter(ImageFilter.GaussianBlur(2)), "10:00:00"),
                (view, "10:00:01"),
                (view.filter(ImageFilter.GaussianBlur(1)), "10:00:02"),
                (night, "21:00:00"),
                (night.filter(ImageFilter.GaussianBlur(2)), "21:00:03"),
            ]
        )
    ]
    import_bytes(
        client, *[(name, jpeg_with_exif(image=img, taken=when)) for name, img, when in shots]
    )
    found = client.post("/api/library/search", json={"query": "the best 2 from july"}).json()
    best = [p["id"] for p in found["photos"]]
    assert sorted(p["filename"] for p in found["photos"]) == ["IMG_1.jpg", "IMG_3.jpg"]

    model = FakeModel([tool("exposure", {"stops": 0.2})], [text("Done.")])
    app.dependency_overrides[get_model] = lambda: model
    run_batch(client, ids=best, instruction="make these consistent")
    assert all(ops_of(client, doc_id) for doc_id in best)

    res = client.post("/api/library/export", json={"ids": best})
    with zipfile.ZipFile(io.BytesIO(res.content)) as archive:
        assert sorted(archive.namelist()) == ["IMG_1-edited.jpg", "IMG_3-edited.jpg"]
