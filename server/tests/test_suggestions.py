from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import pytest
from fake_advisor import FakeAdvisor
from fastapi.testclient import TestClient

from photo_agent import suggestions
from photo_agent.advisor import AdvisorError
from photo_agent.layers import EditState
from photo_agent.main import app
from photo_agent.routes import get_advisor

Upload = Callable[[str], dict[str, Any]]

ZERO = dict.fromkeys(suggestions.SLIDERS, 0)


def direction(title: str, look: str = "none", **sliders: float) -> dict[str, Any]:
    return {
        "title": title,
        "description": f"{title}, described.",
        "look": look,
        "look_strength": 70,
        "sliders": {**ZERO, **sliders},
    }


@pytest.fixture
def advisor() -> Iterator[FakeAdvisor]:
    fake = FakeAdvisor()
    app.dependency_overrides[get_advisor] = lambda: fake
    yield fake
    app.dependency_overrides.pop(get_advisor, None)


def test_sliders_become_operations_and_are_clamped() -> None:
    ops = suggestions.slider_operations(
        {"exposure": 9, "temperature": 20, "contrast": 0, "grain": -5, "vibrance": 30}
    )
    assert [op.summary() for op in ops] == [
        "Exposure (stops +2)",
        "White balance (temperature +20, tint +0)",
        "Vibrance (amount +30)",
    ]


def test_a_direction_with_a_look_adds_the_look_and_a_tuning_layer() -> None:
    made = suggestions.to_suggestion(
        suggestions.Direction.model_validate(direction("Faded film", "film-70s", shadows=10))
    )
    assert made is not None
    look, tuning = made.layers
    assert (look.name, look.opacity) == ("1970s film", 70)
    assert tuning.name == "Faded film: tuning"
    assert suggestions.to_suggestion(suggestions.Direction(title="x", description="")) is None


def test_built_in_directions_fix_a_dark_photo() -> None:
    dark = np.full((40, 60, 3), 0.12, np.float32)
    natural = suggestions.built_in_directions(dark)[0]
    assert natural.sliders["exposure"] > 0.5
    assert natural.sliders["shadows"] > 0


def test_applying_gives_fresh_ids() -> None:
    made = suggestions.suggestions_from(
        [suggestions.Direction.model_validate(direction("Warm", temperature=20))]
    )[0]
    state = suggestions.apply(made, EditState())
    again = suggestions.apply(made, state)
    assert len({layer.id for layer in again.layers}) == 2


def test_built_in_suggestions_without_a_key(client: TestClient, upload: Upload) -> None:
    doc = upload("landscape.png")
    res = client.post(f"/api/documents/{doc['id']}/suggestions")
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["source"] == "built-in"
    assert 2 <= len(body["suggestions"]) <= 4
    first = body["suggestions"][0]
    thumb = client.get(f"/api/documents/{doc['id']}/suggestions/{first['id']}/preview")
    assert thumb.headers["content-type"] == "image/jpeg"
    # The same edits give the same suggestions without asking again.
    assert client.post(f"/api/documents/{doc['id']}/suggestions").json() == body


def test_claude_suggestions_and_picking_one(
    client: TestClient, upload: Upload, advisor: FakeAdvisor
) -> None:
    advisor.answers = [
        {
            "directions": [
                direction("Golden hour", temperature=25, shadows=15),
                direction("Moody", "noir"),
                direction("Nothing at all"),
            ]
        }
    ]
    doc = upload("landscape.png")
    body = client.post(f"/api/documents/{doc['id']}/suggestions").json()
    assert body["source"] == "claude"
    assert [s["title"] for s in body["suggestions"]] == ["Golden hour", "Moody"]
    (call,) = advisor.calls
    assert call["content"][0]["type"] == "image"
    assert "average brightness" in call["content"][1]["text"]

    picked = body["suggestions"][0]
    res = client.post(f"/api/documents/{doc['id']}/suggestions/{picked['id']}")
    assert res.status_code == 200, res.text
    after = res.json()
    assert [layer["name"] for layer in after["state"]["layers"]] == ["Golden hour"]
    assert after["history"][0]["label"] == "Golden hour"
    assert [(e["role"], e["text"]) for e in after["chat"]] == [
        ("user", "Try the “Golden hour” suggestion."),
        ("assistant", "Golden hour, described."),
    ]
    assert after["chat"][1]["step_id"] == after["head"]


def test_refresh_asks_again(client: TestClient, upload: Upload, advisor: FakeAdvisor) -> None:
    advisor.answers = [{"directions": [direction("A", contrast=10)]}] * 2
    doc = upload("landscape.png")
    client.post(f"/api/documents/{doc['id']}/suggestions")
    client.post(f"/api/documents/{doc['id']}/suggestions", json={"refresh": True})
    assert len(advisor.calls) == 2


def test_claude_failure_is_reported(
    client: TestClient, upload: Upload, advisor: FakeAdvisor
) -> None:
    advisor.answers = [AdvisorError("Claude declined to look at this photo.")]
    doc = upload("landscape.png")
    res = client.post(f"/api/documents/{doc['id']}/suggestions")
    assert res.status_code == 502
    assert res.json()["detail"] == "Claude declined to look at this photo."


def test_unknown_suggestion_is_404(client: TestClient, upload: Upload) -> None:
    doc = upload("landscape.png")
    assert client.post(f"/api/documents/{doc['id']}/suggestions/nope").status_code == 404
