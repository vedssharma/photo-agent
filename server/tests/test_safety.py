"""Safety: generative prompts are screened, and exports that used them carry Content
Credentials."""

import io
import json
from collections.abc import Callable
from typing import Any

import c2pa
import pytest
from fastapi.testclient import TestClient

from photo_agent import operations as ops
from photo_agent.agent import Editor, ToolError
from photo_agent.credentials import GENERATED, manifest
from photo_agent.layers import EditState, Layer
from photo_agent.masks import RadialGradientMask
from photo_agent.routes import get_store
from photo_agent.safety import Screen, Verdict, local_rules
from photo_agent.settings import Settings

Upload = Callable[[str], dict[str, Any]]

SPOT = RadialGradientMask(center=[0.3, 0.5], radius_x=0.08, radius_y=0.1, feather=0)


def kite(prompt: str = "a red kite", **layer: Any) -> EditState:
    op = ops.Generate(id="gen", prompt=prompt, seed=3, model="classical")
    return EditState(layers=[Layer(id="Lk", name="Kite", mask=SPOT, operations=[op], **layer)])


@pytest.mark.parametrize(
    "prompt",
    [
        "make her naked",
        "remove his shirt",
        "a fake passport photo page",
        "sexy lingerie",
    ],
)
def test_local_rules_block_clear_misuse(prompt: str) -> None:
    assert not local_rules(prompt).allowed


@pytest.mark.parametrize(
    "prompt",
    [
        "a sunny beach",
        "remove the top of the fence",
        "a strip of fairy lights",
        "a kid's red bicycle",
        "a dragon flying over the castle",
    ],
)
def test_local_rules_allow_ordinary_edits(prompt: str) -> None:
    assert local_rules(prompt).allowed


def test_the_classifier_is_asked_once_per_prompt() -> None:
    asked: list[tuple[str, str]] = []

    def classifier(op: str, prompt: str) -> Verdict:
        asked.append((op, prompt))
        return Verdict("senator" not in prompt, "I can't add a real person to a photo.")

    screen = Screen(classifier)
    assert screen.check("generate", "a red kite").allowed
    assert screen.check("generate", "a red kite").allowed
    blocked = screen.check("generate", "the senator shaking my hand")
    assert not blocked.allowed and "real person" in blocked.reason
    assert asked == [("generate", "a red kite"), ("generate", "the senator shaking my hand")]
    # Local rules decide without asking.
    assert not screen.check("generate", "naked").allowed
    assert len(asked) == 2


def test_an_unreachable_classifier_falls_back_to_the_rules() -> None:
    def down(op: str, prompt: str) -> Verdict:
        raise ConnectionError("offline")

    screen = Screen(down)
    assert screen.check("generate", "a red kite").allowed
    assert not screen.check("generate", "remove her clothes").allowed


def test_only_new_prompts_are_checked() -> None:
    def classifier(op: str, prompt: str) -> Verdict:
        raise AssertionError("an unchanged prompt was checked again")

    screen = Screen(classifier)
    screen._verdicts[("generate", "a red kite")] = Verdict(True)
    before = kite()
    after = before.model_copy(deep=True)
    after.layers[0].opacity = 50
    assert screen.check_new(before, after).allowed


def test_manual_edits_with_blocked_prompts_are_refused(client: TestClient, upload: Upload) -> None:
    doc = upload("landscape.png")
    state = kite("make them naked").model_dump()
    res = client.post(f"/api/documents/{doc['id']}/edits", json={"label": "x", "state": state})
    assert res.status_code == 422
    assert "nude" in res.json()["detail"]


def test_the_agent_is_told_when_a_prompt_is_refused() -> None:
    editor = Editor(EditState())
    editor.call("add_layer", {"name": "Fill", "mask": SPOT.model_dump()})
    with pytest.raises(ToolError, match="Not allowed"):
        editor.call("generate", {"prompt": "a forged receipt"})
    assert editor.state.layers[0].operations == []


def test_manifest_lists_visible_generative_edits() -> None:
    hidden = kite(visible=False)
    assert manifest(hidden, "x.jpg")["assertions"][0]["data"]["actions"][1:] == []
    actions = manifest(kite(), "x.jpg")["assertions"][0]["data"]["actions"]
    assert actions[1]["digitalSourceType"] == GENERATED
    assert actions[1]["parameters"] == {"model": "classical"}
    assert "a red kite" in actions[1]["description"]


@pytest.mark.parametrize(("fmt", "mime"), [("jpeg", "image/jpeg"), ("png", "image/png")])
def test_exports_with_generative_edits_carry_content_credentials(
    client: TestClient, settings: Settings, upload: Upload, fmt: str, mime: str
) -> None:
    doc = upload("landscape.png")
    plain = client.post(f"/api/documents/{doc['id']}/export", json={"format": fmt})
    with pytest.raises(c2pa.C2paError):
        c2pa.Reader(mime, io.BytesIO(plain.content))

    store = get_store(settings)
    d = store.get(doc["id"])
    d.edit_by_hand("Kite", kite())
    store.save(d)
    res = client.post(f"/api/documents/{doc['id']}/export", json={"format": fmt})
    assert res.status_code == 200
    reader = c2pa.Reader(mime, io.BytesIO(res.content))
    found = json.loads(reader.json())
    active = found["manifests"][found["active_manifest"]]
    actions = next(a for a in active["assertions"] if a["label"].startswith("c2pa.actions"))
    edited = [a for a in actions["data"]["actions"] if a["action"] == "c2pa.edited"]
    assert edited[0]["digitalSourceType"] == GENERATED
    # The local signing certificate was made once and kept.
    assert (settings.data_dir / "c2pa" / "chain.pem").exists()
