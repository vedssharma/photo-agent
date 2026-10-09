import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fake_model import FakeModel, text, tool
from fastapi.testclient import TestClient

from photo_agent import references
from photo_agent.agent import AgentEvent, AgentService, history_messages
from photo_agent.operations import MatchReference
from photo_agent.render import RenderCache
from photo_agent.routes import get_store
from photo_agent.settings import Settings

Upload = Callable[[str], dict[str, Any]]


async def _ignore(event: AgentEvent) -> None:
    pass


def share(client: TestClient, doc_id: str, photos: Path, name: str = "low-light.jpg") -> Any:
    res = client.post(
        f"/api/documents/{doc_id}/references",
        files={"file": (name, (photos / name).read_bytes())},
    )
    assert res.status_code == 201, res.text
    return res.json()


def test_sharing_and_matching_a_reference(client: TestClient, upload: Upload, photos: Path) -> None:
    doc = upload("landscape.png")
    ref = share(client, doc["id"], photos)
    assert ref["filename"] == "low-light.jpg"
    assert len(ref["stats"]["tone"]) == 17
    image = client.get(f"/api/documents/{doc['id']}/references/{ref['id']}")
    assert image.headers["content-type"] == "image/jpeg"

    before = client.get(f"/api/documents/{doc['id']}/preview").content
    after = client.post(f"/api/documents/{doc['id']}/references/{ref['id']}/match").json()
    (layer,) = after["state"]["layers"]
    assert layer["name"] == "Match low-light.jpg"
    assert layer["operations"][0]["stats"] == ref["stats"]
    assert client.get(f"/api/documents/{doc['id']}/preview").content != before


def test_unknown_or_bad_references(client: TestClient, upload: Upload) -> None:
    doc = upload("landscape.png")
    assert client.get(f"/api/documents/{doc['id']}/references/rnope").status_code == 404
    assert client.post(f"/api/documents/{doc['id']}/references/rnope/match").status_code == 404
    res = client.post(
        f"/api/documents/{doc['id']}/references", files={"file": ("x.jpg", b"not a photo")}
    )
    assert res.status_code == 415


def test_hand_edits_get_the_measurements_filled_in(
    client: TestClient, upload: Upload, photos: Path
) -> None:
    doc = upload("landscape.png")
    ref = share(client, doc["id"], photos)
    layer: dict[str, Any] = {
        "id": "Lm",
        "name": "Match",
        "operations": [{"op": "match_reference", "reference": ref["id"], "id": "m1"}],
    }
    res = client.post(
        f"/api/documents/{doc['id']}/edits",
        json={"label": "Match", "state": {"framing": [], "layers": [layer]}},
    )
    assert res.json()["state"]["layers"][0]["operations"][0]["stats"] == ref["stats"]
    layer["operations"][0]["reference"] = "rnope"
    res = client.post(
        f"/api/documents/{doc['id']}/edits",
        json={"label": "Match", "state": {"framing": [], "layers": [layer]}},
    )
    assert res.status_code == 422


def test_agent_sees_shared_references_and_matches_them(
    client: TestClient, upload: Upload, photos: Path, settings: Settings
) -> None:
    doc = upload("landscape.png")
    ref = share(client, doc["id"], photos)
    model = FakeModel(
        [
            tool("add_layer", {"name": "Like the night shot"}),
            tool("match_reference", {"reference": ref["id"], "color": 70}),
        ],
        [text("Matched its moody colors.")],
    )
    service = AgentService(get_store(settings), RenderCache(), model)
    result = asyncio.run(
        service.run_turn(doc["id"], "make it look like this", _ignore, shared=[ref["id"], "rnope"])
    )
    content = model.calls[0][-1]["content"]
    assert [b["type"] for b in content] == ["image", "text", "image", "text"]
    assert content[1]["text"] == f"Reference photo {ref['id']} (low-light.jpg):"
    (op,) = result.state.layers[0].operations
    assert isinstance(op, MatchReference) and op.stats is not None and op.color == 70
    assert result.chat[0].references == [ref["id"]]
    # The tool result and later replays keep the measurements out of Claude's way.
    check = model.calls[1][-1]["content"][0]["content"]
    assert "lab_mean" not in str(check)
    replay, _ = history_messages(result.chat)
    assert f"(Shared reference photos: {ref['id']}.)" in str(replay[0]["content"])


def test_matching_an_unknown_reference_is_an_error(upload: Upload, settings: Settings) -> None:
    doc = upload("landscape.png")
    model = FakeModel([tool("match_reference", {"reference": "rnope"})], [text("Sorry.")])
    service = AgentService(get_store(settings), RenderCache(), model)
    asyncio.run(service.run_turn(doc["id"], "like that", _ignore))
    error = model.calls[1][-1]["content"][0]
    assert error["is_error"] and "No reference photo" in error["content"]


def test_fill_stats_leaves_complete_states_alone(tmp_path: Path) -> None:
    from photo_agent.layers import EditState

    state = EditState()
    assert references.fill_stats(state, tmp_path) is state
