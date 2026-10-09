import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fake_model import FakeModel, text, tool
from fastapi.testclient import TestClient

from photo_agent import style
from photo_agent.agent import AgentEvent, AgentService
from photo_agent.layers import EditState, Layer
from photo_agent.operations import Contrast, Exposure, Grain, WhiteBalance
from photo_agent.render import RenderCache
from photo_agent.routes import get_store
from photo_agent.settings import Settings

Upload = Callable[[str], dict[str, Any]]


def state(*ops: Any, opacity: float = 100, mask: Any = None) -> EditState:
    return EditState(layers=[Layer(id="L1", name="x", opacity=opacity, mask=mask, operations=ops)])


def test_slider_values_weigh_opacity_and_skip_masked_layers() -> None:
    s = state(WhiteBalance(temperature=20, tint=-4), Contrast(amount=10), opacity=50)
    assert style.slider_values(s) == {"temperature": 10, "tint": -2, "contrast": 5}
    masked = state(
        Contrast(amount=30),
        mask={"kind": "radial", "center": [0.5, 0.5], "radius_x": 0.3, "radius_y": 0.3},
    )
    assert style.slider_values(masked) == {}


def test_tendencies_and_usual_look(tmp_path: Path) -> None:
    taste = style.StyleStore(tmp_path)
    assert taste.load().describe() is None
    assert taste.load().usual_look() is None
    taste.kept(state(WhiteBalance(temperature=10), Grain(amount=20), Exposure(stops=0.5)))
    taste.kept(state(WhiteBalance(temperature=20), Exposure(stops=0.3)))
    taste.adjusted(state(Grain(amount=50)), state(Grain(amount=30)))
    profile = taste.load()
    found = {t.slider: t for t in profile.tendencies()}
    assert found["temperature"].preferred == 15
    assert found["grain"].preferred == round((20 + 30 * 2) / 3, 2)
    described = profile.describe()
    assert described is not None and "warmer (temperature around +15" in described
    look = profile.usual_look()
    assert look is not None and look.name == "My usual look"
    # Exposure depends on the photo, so the usual look leaves it out.
    assert sorted(op.op for op in look.operations) == ["grain", "white_balance"]


def test_rejections_are_described_but_not_preferred(tmp_path: Path) -> None:
    taste = style.StyleStore(tmp_path)
    taste.rejected(EditState(), state(Contrast(amount=40)))
    profile = taste.load()
    assert profile.tendencies() == []
    described = profile.describe()
    assert described is not None and "recently undid: contrast +40" in described


def test_drags_of_one_slider_fold_into_one_note(tmp_path: Path) -> None:
    taste = style.StyleStore(tmp_path)
    for amount in (5, 10, 15):
        taste.adjusted(EditState(), state(Contrast(amount=amount)))
    assert [s.value for s in taste.load().signals] == [15]


def test_routes_learn_from_download_undo_and_hand_edits(
    client: TestClient, upload: Upload, settings: Settings
) -> None:
    doc = upload("portrait.jpg")
    warm = state(WhiteBalance(temperature=18)).model_dump()
    client.post(f"/api/documents/{doc['id']}/edits", json={"label": "Warm", "state": warm})
    client.post(f"/api/documents/{doc['id']}/export", json={"format": "jpeg"})
    summary = client.get("/api/style").json()
    assert summary["signals"] == 2  # set by hand, then kept
    assert summary["has_usual_look"] is True
    assert summary["lines"] == ["warmer (temperature around +18, 2 edits)"]

    # An undone agent step counts against what it added.
    asyncio.run(
        AgentService(get_store(settings), RenderCache(), FakeModel([text("x")])).run_turn(
            doc["id"], "hi", _ignore
        )
    )
    model = FakeModel([tool("saturation", {"amount": 60})], [text("More color.")])
    asyncio.run(
        AgentService(get_store(settings), RenderCache(), model).run_turn(
            doc["id"], "more color", _ignore
        )
    )
    client.post(f"/api/documents/{doc['id']}/undo")
    profile = style.StyleStore(settings.data_dir).load()
    assert profile.signals[-1].model_dump(include={"slider", "value", "verdict"}) == {
        "slider": "saturation",
        "value": 60,
        "verdict": "rejected",
    }

    res = client.post(f"/api/documents/{doc['id']}/style/usual-look")
    assert res.status_code == 200
    assert res.json()["state"]["layers"][-1]["name"] == "My usual look"

    assert client.delete("/api/style").status_code == 204
    assert client.get("/api/style").json()["signals"] == 0
    assert client.post(f"/api/documents/{doc['id']}/style/usual-look").status_code == 409


async def _ignore(event: AgentEvent) -> None:
    pass


def test_agent_hears_the_taste_and_can_apply_the_usual_look(
    upload: Upload, settings: Settings
) -> None:
    taste = style.StyleStore(settings.data_dir)
    taste.kept(state(WhiteBalance(temperature=12)))
    taste.kept(state(WhiteBalance(temperature=16)))
    doc = upload("portrait.jpg")
    model = FakeModel([tool("apply_usual_look", {})], [text("Your usual look.")])
    service = AgentService(get_store(settings), RenderCache(), model, taste=taste)
    result = asyncio.run(service.run_turn(doc["id"], "my usual look please", _ignore))
    asked = model.calls[0][-1]["content"][1]["text"]
    assert "tends toward warmer" in asked
    assert [layer.name for layer in result.state.layers] == ["My usual look"]


def test_style_memory_can_be_turned_off(client: TestClient, settings: Settings) -> None:
    settings.style_memory = False
    assert client.get("/api/style").json()["signals"] == 0
