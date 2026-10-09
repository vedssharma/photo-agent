from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import pytest
from fake_advisor import FakeAdvisor
from fastapi.testclient import TestClient

from photo_agent import critique
from photo_agent.agent import history_messages
from photo_agent.graph import Document
from photo_agent.main import app
from photo_agent.routes import get_advisor

Upload = Callable[[str], dict[str, Any]]


@pytest.fixture
def advisor() -> Iterator[FakeAdvisor]:
    fake = FakeAdvisor()
    app.dependency_overrides[get_advisor] = lambda: fake
    yield fake
    app.dependency_overrides.pop(get_advisor, None)


def test_built_in_critique_flags_a_dark_flat_yellow_photo() -> None:
    dark = np.zeros((40, 60, 3), np.float32)
    dark[...] = (0.2, 0.17, 0.05)
    found = critique.built_in(dark)
    improve = {p.aspect: p for p in found.points if p.verdict == "improve"}
    assert {"exposure", "mood", "color"} <= set(improve)
    assert improve["exposure"].fix and improve["exposure"].fix_label == "Brighten it"
    assert "worth fixing" in found.summary


def test_built_in_critique_praises_a_balanced_photo() -> None:
    rng = np.random.default_rng(1)
    photo = rng.uniform(0.05, 0.9, (40, 60, 1)).astype(np.float32).repeat(3, axis=2)
    found = critique.built_in(photo)
    assert all(p.verdict == "good" for p in found.points)
    assert all(p.fix is None for p in found.points)


def test_critique_without_a_key_goes_into_the_chat(client: TestClient, upload: Upload) -> None:
    doc = upload("low-light.jpg")
    res = client.post(f"/api/documents/{doc['id']}/critique")
    assert res.status_code == 200, res.text
    user, reply = res.json()["chat"]
    assert user["text"] == "What do you think of this photo?"
    assert reply["critique"]["source"] == "built-in"
    assert reply["text"] == reply["critique"]["summary"]


def test_claude_critique_is_tidied(
    client: TestClient, upload: Upload, advisor: FakeAdvisor
) -> None:
    advisor.answers = [
        {
            "summary": "A lovely candid moment.",
            "points": [
                {
                    "aspect": "subject",
                    "verdict": "good",
                    "text": "Her smile carries it.",
                    "fix": "make it better",
                    "fix_label": "x",
                },
                {
                    "aspect": "composition",
                    "verdict": "improve",
                    "text": "The horizon cuts through her head.",
                    "fix": "Crop a little lower so the horizon sits below her shoulders",
                    "fix_label": None,
                },
            ],
        }
    ]
    doc = upload("portrait.jpg")
    body = client.post(f"/api/documents/{doc['id']}/critique").json()
    good, improve = body["chat"][-1]["critique"]["points"]
    assert good["fix"] is None and good["fix_label"] is None
    assert improve["fix_label"] == "Fix it"
    assert advisor.calls[0]["content"][0]["type"] == "image"

    # The agent hears what it said, fixes included.
    stored = Document.model_validate(client.get(f"/api/documents/{doc['id']}/graph").json())
    messages, _ = history_messages(stored.chat)
    said = str(messages[-1]["content"])
    assert "composition, improve: The horizon" in said
    assert "suggested fix: Crop a little lower" in said
