import asyncio
from collections.abc import Callable
from typing import Any

import pytest
from fake_model import FakeModel, text, tool
from fastapi.testclient import TestClient

from photo_agent.agent import AgentEvent, AgentService, Editor, ToolError, history_messages
from photo_agent.graph import ChatEntry, Document, Plan, PlanStep
from photo_agent.layers import EditState
from photo_agent.render import RenderCache
from photo_agent.routes import get_model, get_store
from photo_agent.settings import Settings

Upload = Callable[[str], dict[str, Any]]

PLAN = {
    "steps": [
        {"text": "Remove the people in the background", "kind": "ai"},
        {"text": "Replace the sky with a sunset", "kind": "generative"},
        {"text": "Warm up the colors", "kind": "adjust"},
    ]
}
SKY = {"kind": "semantic", "target": "sky"}


def run_turn(
    settings: Settings, doc_id: str, request: str, model: FakeModel, approve: bool = False
) -> Any:
    async def emit(event: AgentEvent) -> None:
        pass

    service = AgentService(get_store(settings), RenderCache(), model)
    return asyncio.run(service.run_turn(doc_id, request, emit, approve_plan=approve))


def test_proposing_a_plan_ends_the_turn_and_waits(upload: Upload, settings: Settings) -> None:
    doc = upload("landscape.png")
    model = FakeModel([text("Here's what I'd do."), tool("propose_plan", PLAN)])
    result = run_turn(settings, doc["id"], "clean it up and give it a sunset sky", model)
    assert len(model.calls) == 1  # no second round after the plan
    assert result.steps == []
    plan = result.pending_plan
    assert plan is not None and [s.kind for s in plan.steps] == ["ai", "generative", "adjust"]
    assert result.chat[-1].text == "Here's what I'd do."

    # Going ahead runs it, with the plan in front of Claude, and allows several generative
    # steps; afterwards nothing is pending.
    go = FakeModel(
        [
            tool("add_layer", {"name": "Sunset sky", "mask": SKY}),
            tool("generate", {"prompt": "a sunset sky"}),
            tool("relight", {"prompt": "warm sunset light from the left"}),
        ],
        [text("Done: new sky and warmer light.")],
    )
    after = run_turn(settings, doc["id"], "Go ahead", go, approve=True)
    asked = go.calls[0][-1]["content"][1]["text"]
    assert "Approved plan:\n1. Remove the people in the background" in asked
    assert asked.endswith("3. Warm up the colors\n\nGo ahead")
    assert [lay.name for lay in after.state.layers][:1] == ["Sunset sky"]
    assert len(after.state.layers) == 2
    assert after.pending_plan is None
    # The plan is replayed to Claude in later turns.
    replay, _ = history_messages(after.chat)
    assert "Proposed plan:\n1. Remove" in str(replay[1]["content"])


def test_typing_something_else_drops_the_plan(upload: Upload, settings: Settings) -> None:
    doc = upload("landscape.png")
    run_turn(settings, doc["id"], "do a lot", FakeModel([tool("propose_plan", PLAN)]))
    model = FakeModel([text("Sure, skipping the sky.")])
    # Even if the browser says "approve", a plan that is no longer pending is not approved.
    result = run_turn(settings, doc["id"], "skip the sky", model)
    assert result.pending_plan is None
    late = run_turn(settings, doc["id"], "Go ahead", FakeModel([text("OK.")]), approve=True)
    assert "Approved plan" not in str(late.chat)


def test_second_generative_edit_needs_a_plan() -> None:
    editor = Editor(EditState())
    editor.call("relight", {"prompt": "golden hour"})
    with pytest.raises(ToolError, match="propose_plan"):
        editor.call("restyle", {"prompt": "watercolor"})
    assert len(editor.state.layers) == 1
    editor.plan_approved = True
    editor.call("restyle", {"prompt": "watercolor"})
    assert len(editor.state.layers) == 2


def test_invalid_plan_is_an_error() -> None:
    with pytest.raises(ToolError):
        Editor(EditState()).call("propose_plan", {"steps": []})


def test_pending_plan_is_the_last_reply_only() -> None:
    plan = Plan(steps=[PlanStep(text="x")])
    entries = [ChatEntry(role="user", text="a"), ChatEntry(role="assistant", text="b", plan=plan)]
    doc = Document(filename="x.jpg", format="JPEG", width=1, height=1, chat=entries)
    assert doc.pending_plan == plan
    doc.chat.append(ChatEntry(role="event", text="Edited by hand: Exposure"))
    assert doc.pending_plan == plan
    doc.chat.append(ChatEntry(role="user", text="c"))
    assert doc.pending_plan is None


def test_websocket_passes_the_go_ahead(client: TestClient, upload: Upload) -> None:
    doc = upload("landscape.png")
    model = FakeModel([tool("propose_plan", PLAN)], [text("Doing it.")])
    client.app.dependency_overrides[get_model] = lambda: model  # type: ignore[attr-defined]

    def turn(ws: Any, message: dict[str, Any]) -> dict[str, Any]:
        ws.send_json(message)
        while True:
            event: dict[str, Any] = ws.receive_json()
            if event["type"] in ("done", "error"):
                return event

    with client.websocket_connect(f"/api/documents/{doc['id']}/chat") as ws:
        done = turn(ws, {"type": "message", "text": "lots"})
        assert done["document"]["pending_plan"]["steps"][0]["kind"] == "ai"
        done = turn(ws, {"type": "message", "text": "Go ahead", "approve_plan": True})
    assert done["document"]["pending_plan"] is None
    assert "Approved plan" in model.calls[1][-1]["content"][1]["text"]
