from collections.abc import Callable
from typing import Any

from fastapi.testclient import TestClient

from photo_agent.layers import EditState, Layer
from photo_agent.masks import BrushMask, BrushStroke, LuminosityMask
from photo_agent.operations import Contrast, Crop, Exposure
from photo_agent.recipes import Recipe, apply, portable

Upload = Callable[[str], dict[str, Any]]

MOODY: dict[str, Any] = {
    "id": "Lmoody00",
    "name": "Moody",
    "blend_mode": "luminosity",
    "opacity": 80,
    "operations": [{"id": "c1", "op": "contrast", "amount": 30}],
}


def test_portable_drops_brush_masks_only() -> None:
    brushed = Layer(
        id="La",
        name="Face",
        mask=BrushMask(strokes=[BrushStroke(points=[(0.5, 0.5)], size=0.1)]),
        operations=[Exposure(stops=0.3)],
    )
    toned = Layer(id="Lb", name="Sky", mask=LuminosityMask(low=0.7, high=1))
    kept = portable([brushed, toned])
    assert kept[0].mask is None
    assert kept[0].operations == brushed.operations
    assert isinstance(kept[1].mask, LuminosityMask)
    assert brushed.mask is not None  # The originals are untouched.


def test_apply_adds_layers_on_top_with_fresh_ids() -> None:
    recipe = Recipe.model_validate(
        {"id": "r", "name": "Moody", "created_at": "2026-10-08T00:00:00Z", "layers": [MOODY]}
    )
    state = EditState(
        framing=[Crop(aspect="1:1", left=0, top=0, right=1, bottom=1)],
        layers=[Layer(id="Lbase", name="Base", operations=[Contrast(amount=5)])],
    )
    after = apply(recipe, state)
    assert after.framing == state.framing
    assert [layer.name for layer in after.layers] == ["Base", "Moody"]
    added = after.layers[1]
    assert added.id != "Lmoody00"
    assert added.operations[0].id != "c1"
    assert added.opacity == 80
    assert recipe.layers[0].id == "Lmoody00"


def test_recipe_api_round_trip(client: TestClient, upload: Upload) -> None:
    assert client.get("/api/recipes").json() == []
    res = client.post("/api/recipes", json={"name": "  Moody ", "layers": [MOODY]})
    assert res.status_code == 201, res.text
    recipe = res.json()
    assert recipe["name"] == "Moody"
    assert [r["id"] for r in client.get("/api/recipes").json()] == [recipe["id"]]

    doc = upload("portrait.jpg")
    applied = client.post(f"/api/documents/{doc['id']}/recipes/{recipe['id']}")
    assert applied.status_code == 200, applied.text
    body = applied.json()
    assert [s["label"] for s in body["history"]] == ["Apply recipe “Moody”"]
    assert body["state"]["layers"][0]["name"] == "Moody"
    assert body["revision"] != "original"

    assert client.post(f"/api/documents/{doc['id']}/recipes/nope").status_code == 404
    assert client.delete(f"/api/recipes/{recipe['id']}").status_code == 204
    assert client.get("/api/recipes").json() == []
    assert client.delete(f"/api/recipes/{recipe['id']}").status_code == 404


def test_recipe_needs_a_name_and_layers(client: TestClient) -> None:
    assert client.post("/api/recipes", json={"name": "  ", "layers": [MOODY]}).status_code == 422
    assert client.post("/api/recipes", json={"name": "x", "layers": []}).status_code == 422
