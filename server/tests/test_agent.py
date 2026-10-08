import asyncio
from collections.abc import Callable
from typing import Any

import pytest
from fake_model import FakeModel, text, tool
from fastapi.testclient import TestClient

from photo_agent.agent import (
    TOOLS,
    AgentEvent,
    AgentService,
    Editor,
    ToolError,
    history_messages,
)
from photo_agent.graph import ChatEntry
from photo_agent.operations import OPERATIONS_BY_NAME, Exposure
from photo_agent.render import RenderCache
from photo_agent.routes import get_model, get_store
from photo_agent.settings import Settings

Upload = Callable[[str], dict[str, Any]]


def run_turn(
    settings: Settings, doc_id: str, request: str, model: FakeModel
) -> tuple[list[AgentEvent], Any]:
    events: list[AgentEvent] = []

    async def emit(event: AgentEvent) -> None:
        events.append(event)

    service = AgentService(get_store(settings), RenderCache(), model)
    doc = asyncio.run(service.run_turn(doc_id, request, emit))
    return events, doc


def test_every_operation_is_a_tool() -> None:
    names = {t["name"] for t in TOOLS}
    assert set(OPERATIONS_BY_NAME) <= names
    assert {"update_operation", "remove_operation"} <= names
    exposure: dict[str, Any] = dict(next(t for t in TOOLS if t["name"] == "exposure"))
    assert "id" not in exposure["input_schema"]["properties"]
    assert exposure["input_schema"]["required"] == ["stops"]


def test_editor_adds_updates_and_removes() -> None:
    editor = Editor([])
    editor.call("exposure", {"stops": 0.5})
    op_id = editor.operations[0].id
    editor.call("update_operation", {"id": op_id, "changes": {"stops": 0.3}})
    assert editor.operations == [Exposure(id=op_id, stops=0.3)]
    editor.call("remove_operation", {"id": op_id})
    assert editor.operations == []


@pytest.mark.parametrize(
    ("name", "args"),
    [
        ("exposure", {"stops": 12}),
        ("exposure", {}),
        ("teleport", {}),
        ("update_operation", {"id": "nope", "changes": {}}),
        ("exposure", "not an object"),
    ],
)
def test_editor_rejects_bad_calls(name: str, args: Any) -> None:
    with pytest.raises(ToolError):
        Editor([]).call(name, args)


def test_turn_records_operations_and_reply(upload: Upload, settings: Settings) -> None:
    doc = upload("phone.heic")
    model = FakeModel(
        [
            text("Warming it up."),
            tool("white_balance", {"temperature": 25}),
            tool("crop", {"aspect": "4:5"}),
        ],
        [text("I warmed the colors and cropped it for Instagram.")],
    )
    events, result = run_turn(settings, doc["id"], "warmer, and crop for instagram", model)

    assert [type(e).__name__ for e in events] == [
        "TurnStarted",
        "TextDelta",
        "OperationEvent",
        "OperationEvent",
        "TextDelta",
        "TextDelta",
        "TurnDone",
    ]
    assert [op.op for op in result.operations] == ["white_balance", "crop"]
    assert result.turns[0].reply == (
        "Warming it up.\n\nI warmed the colors and cropped it for Instagram."
    )
    assert [c.role for c in result.chat] == ["user", "assistant"]
    assert result.chat[1].turn_id == result.turns[0].id

    # Claude saw the photo and the (empty) operation list with the request.
    first_request = model.calls[0][-1]["content"]
    assert first_request[0]["type"] == "image"
    assert "No edits yet" in first_request[1]["text"]
    # And got tool results back in a single user message.
    results = model.calls[1][-1]["content"]
    assert [r["type"] for r in results] == ["tool_result", "tool_result"]


def test_invalid_tool_input_goes_back_to_claude_as_an_error(
    upload: Upload, settings: Settings
) -> None:
    doc = upload("portrait.jpg")
    model = FakeModel(
        [tool("exposure", {"stops": 40})],
        [tool("exposure", {"stops": 0.4}, id="toolu_2")],
        [text("Brightened a little.")],
    )
    _, result = run_turn(settings, doc["id"], "brighter", model)
    error = model.calls[1][-1]["content"][0]
    assert error["is_error"] is True
    assert "stops" in error["content"]
    assert [op.stops for op in result.operations] == [0.4]


def test_reply_without_edits_does_not_create_an_undo_step(
    upload: Upload, settings: Settings
) -> None:
    doc = upload("portrait.jpg")
    model = FakeModel([text("Do you want it square or 4:5?")])
    _, result = run_turn(settings, doc["id"], "crop it", model)
    assert result.turns == []
    assert [c.text for c in result.chat] == ["crop it", "Do you want it square or 4:5?"]


def test_refusal_is_reported_without_edits(upload: Upload, settings: Settings) -> None:
    doc = upload("portrait.jpg")
    model = FakeModel([], stop_reason="refusal")
    model.responses = [[]]
    _, result = run_turn(settings, doc["id"], "something", model)
    assert result.turns == []
    assert "can't help" in result.chat[-1].text


def test_history_replays_conversation_and_undo_events() -> None:
    chat = [
        ChatEntry(role="user", text="warmer"),
        ChatEntry(role="assistant", text="Warmed it."),
        ChatEntry(role="event", text="Undid: warmer"),
    ]
    messages, pending = history_messages(chat)
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert pending == ["Undid: warmer"]


def test_next_request_mentions_undo(upload: Upload, settings: Settings, client: TestClient) -> None:
    doc = upload("portrait.jpg")
    run_turn(
        settings,
        doc["id"],
        "brighter",
        FakeModel([tool("exposure", {"stops": 0.5})], [text("Brighter now.")]),
    )
    client.post(f"/api/documents/{doc['id']}/undo")
    model = FakeModel([text("OK.")])
    run_turn(settings, doc["id"], "hmm", model)
    request_text = model.calls[0][-1]["content"][1]["text"]
    assert "Undid: brighter" in request_text
    assert "No edits yet" in request_text


def test_chat_websocket_streams_a_turn(client: TestClient, upload: Upload) -> None:
    doc = upload("portrait.jpg")
    model = FakeModel([tool("exposure", {"stops": 0.3})], [text("Done.")])
    client.app.dependency_overrides[get_model] = lambda: model  # type: ignore[attr-defined]
    with client.websocket_connect(f"/api/documents/{doc['id']}/chat") as ws:
        ws.send_json({"type": "message", "text": "brighter"})
        events = []
        while True:
            event = ws.receive_json()
            events.append(event)
            if event["type"] in ("done", "error"):
                break
    assert events[0]["type"] == "turn_started"
    assert events[1] == {"type": "operation", "action": "added", "summary": "Exposure (stops +0.3)"}
    assert events[-1]["document"]["operations"][0]["stops"] == 0.3
    assert events[-1]["document"]["can_undo"] is True


def test_chat_websocket_explains_missing_api_key(client: TestClient, upload: Upload) -> None:
    doc = upload("portrait.jpg")
    with client.websocket_connect(f"/api/documents/{doc['id']}/chat") as ws:
        ws.send_json({"type": "message", "text": "brighter"})
        event = ws.receive_json()
    assert event["type"] == "error"
    assert "ANTHROPIC_API_KEY" in event["message"]
