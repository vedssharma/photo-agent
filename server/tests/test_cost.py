import asyncio
from collections.abc import Callable, Iterator
from typing import Any

import pytest
from fake_advisor import FakeAdvisor
from fake_model import FakeModel, text, tool
from fastapi.testclient import TestClient

from photo_agent.agent import AgentEvent, AgentService, ClaudeModel
from photo_agent.main import app
from photo_agent.render import RenderCache
from photo_agent.routes import get_advisor, get_store
from photo_agent.routing import choose_tier
from photo_agent.settings import Settings

Upload = Callable[[str], dict[str, Any]]


async def _ignore(event: AgentEvent) -> None:
    pass


@pytest.mark.parametrize(
    ("request_text", "tier"),
    [
        ("a bit warmer", "routine"),
        ("less contrast please", "routine"),
        ("crop it square", "routine"),
        ("brighter and a touch more color", "routine"),
        ("remove the person on the left", "deep"),
        ("replace the sky with a sunset", "deep"),
        ("what do you think?", "deep"),
        ("make it look like this", "deep"),
        ("warmer, brighter, more contrast, less saturation, and crop it", "deep"),
        (" ".join(["word"] * 30), "deep"),
    ],
)
def test_routine_requests_go_to_the_smaller_model(request_text: str, tier: str) -> None:
    assert choose_tier(request_text) == tier


def test_plans_and_references_always_get_the_main_model() -> None:
    assert choose_tier("go ahead", plan_approved=True) == "deep"
    assert choose_tier("like this", shared_references=True) == "deep"


def test_turns_carry_their_tier_and_escalate_when_struggling(
    upload: Upload, settings: Settings
) -> None:
    doc = upload("portrait.jpg")
    model = FakeModel(
        [tool("exposure", {"stops": 99})],
        [tool("exposure", {"stops": 0.3}, id="t2")],
        [text("Brighter.")],
    )
    service = AgentService(get_store(settings), RenderCache(), model)
    asyncio.run(service.run_turn(doc["id"], "a bit brighter", _ignore))
    # Every call in the first round failed, so the rest of the turn uses the main model.
    assert model.tiers == ["routine", "deep", "deep"]


class FakeStream:
    def __init__(self, calls: list[dict[str, Any]], **kwargs: Any) -> None:
        calls.append(kwargs)

    async def __aenter__(self) -> "FakeStream":
        return self

    async def __aexit__(self, *exc: object) -> None:
        pass

    def __aiter__(self) -> "FakeStream":
        return self

    async def __anext__(self) -> Any:
        raise StopAsyncIteration

    async def get_final_message(self) -> Any:
        from anthropic.types.beta import BetaMessage

        return BetaMessage.model_validate(
            {
                "id": "m",
                "type": "message",
                "role": "assistant",
                "model": "x",
                "content": [],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 10, "output_tokens": 2, "cache_read_input_tokens": 8},
            }
        )


@pytest.mark.parametrize(
    ("tier", "routine_model", "model", "effort", "fallbacks"),
    [
        ("deep", "claude-sonnet-5-5", "claude-opus-5-5", "medium", True),
        ("routine", "claude-sonnet-5-5", "claude-sonnet-5-5", "low", True),
        ("routine", "claude-haiku-5-5", "claude-haiku-5-5", "low", False),
    ],
)
def test_claude_model_picks_model_effort_and_caches_the_prefix(
    tier: str,
    routine_model: str,
    model: str,
    effort: str,
    fallbacks: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    claude = ClaudeModel("sk-test", "claude-opus-5-5", routine_model)
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        claude.client.beta.messages, "stream", lambda **kwargs: FakeStream(calls, **kwargs)
    )

    async def on_text(t: str) -> None:
        pass

    asyncio.run(
        claude.create(system="sys", tools=[], messages=[], on_text=on_text, tier=tier)  # type: ignore[arg-type]
    )
    (sent,) = calls
    assert sent["model"] == model
    assert sent["output_config"] == {"effort": effort}
    assert sent["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert sent["cache_control"] == {"type": "ephemeral"}
    assert ("fallbacks" in sent) is fallbacks


@pytest.fixture
def advisor() -> Iterator[FakeAdvisor]:
    fake = FakeAdvisor()
    fake.model = "claude-test"  # type: ignore[attr-defined]
    app.dependency_overrides[get_advisor] = lambda: fake
    yield fake
    app.dependency_overrides.pop(get_advisor, None)


def test_analyses_of_the_same_edits_are_reused(
    client: TestClient, upload: Upload, advisor: FakeAdvisor
) -> None:
    feedback = {"summary": "Nice.", "points": []}
    advisor.answers = [feedback, feedback]
    doc = upload("portrait.jpg")
    client.post(f"/api/documents/{doc['id']}/critique")
    client.post(f"/api/documents/{doc['id']}/critique")
    assert len(advisor.calls) == 1
    # New edits mean a fresh look; undoing back reuses the first one.
    exposure = {"id": "Lx", "name": "x", "operations": [{"op": "exposure", "stops": 0.4}]}
    client.post(
        f"/api/documents/{doc['id']}/edits",
        json={"label": "x", "state": {"framing": [], "layers": [exposure]}},
    )
    client.post(f"/api/documents/{doc['id']}/critique")
    client.post(f"/api/documents/{doc['id']}/undo")
    after = client.post(f"/api/documents/{doc['id']}/critique").json()
    assert len(advisor.calls) == 2
    assert [e["text"] for e in after["chat"] if e["role"] == "assistant"] == ["Nice."] * 4
