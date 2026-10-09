"""Portrait retouching: skin smoothing, blemish healing, eyes and teeth."""

from collections.abc import Callable
from typing import Any

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from photo_agent import operations as ops
from photo_agent.agent import Editor, ToolError
from photo_agent.layers import EditState
from photo_agent.masks import SemanticMask
from photo_agent.portrait import Retouch, retouch_layers
from photo_agent.render import RenderContext, apply_operations, render_state
from photo_agent.routes import get_store
from photo_agent.settings import Settings

Upload = Callable[[str], dict[str, Any]]
CTX = RenderContext()


def skin_patch(size: int = 200, seed: int = 0) -> np.ndarray:
    """Even skin tone with fine texture, like pores."""
    rng = np.random.default_rng(seed)
    base = np.full((size, size, 3), (0.8, 0.6, 0.5), np.float32)
    base += rng.normal(0, 0.02, (size, size, 1)).astype(np.float32)
    np.clip(base, 0, 1, out=base)
    return base


def band(a: np.ndarray, fine: float, coarse: float) -> float:
    return float((cv2.GaussianBlur(a, (0, 0), fine) - cv2.GaussianBlur(a, (0, 0), coarse)).std())


def pores(a: np.ndarray) -> float:
    return float((a - cv2.GaussianBlur(a, (0, 0), 1)).std())


def test_smooth_skin_evens_tone_but_keeps_texture() -> None:
    skin = skin_patch(512)
    rng = np.random.default_rng(1)
    blotches = cv2.GaussianBlur(rng.normal(0, 1, (512, 512)).astype(np.float32), (0, 0), 3)
    skin += (0.04 * blotches / blotches.std())[..., None]
    out = apply_operations(skin.copy(), [ops.SmoothSkin(amount=100, texture=60)], CTX)
    assert band(out, 2, 8) < 0.6 * band(skin, 2, 8)  # blotchy tone evened out
    assert pores(out) > 0.3 * pores(skin)  # pores are still there
    keep = apply_operations(skin.copy(), [ops.SmoothSkin(amount=100, texture=100)], CTX)
    assert pores(keep) > pores(out)


def test_heal_blemishes_removes_small_spots_only() -> None:
    skin = skin_patch(600)
    cv2.circle(skin, (150, 150), 3, (0.5, 0.3, 0.25), -1)  # a blemish
    cv2.ellipse(skin, (400, 350), (40, 18), 0, 0, 360, (0.3, 0.2, 0.2), -1)  # an eye
    out = apply_operations(skin.copy(), [ops.HealBlemishes(amount=80, size=60)], CTX)
    assert out[147:154, 147:154].mean() > skin[147:154, 147:154].mean() + 0.1
    np.testing.assert_allclose(out[320:380, 340:460], skin[320:380, 340:460], atol=0.01)


def test_retouch_layers_have_face_part_masks() -> None:
    layers = retouch_layers(Retouch())
    assert [layer.name for layer in layers] == ["Smooth skin", "Brighten eyes", "Whiten teeth"]
    assert [layer.mask for layer in layers] == [
        SemanticMask(target="skin"),
        SemanticMask(target="eyes"),
        SemanticMask(target="teeth"),
    ]
    assert [op.op for op in layers[0].operations] == ["heal_blemishes", "smooth_skin"]
    only = retouch_layers(Retouch(smooth_skin=0, brighten_eyes=0, whiten_teeth=0))
    assert [layer.name for layer in only] == ["Heal blemishes"]
    assert (
        retouch_layers(
            Retouch(smooth_skin=0, remove_blemishes=False, brighten_eyes=0, whiten_teeth=0)
        )
        == []
    )


def test_the_agent_retouches_a_portrait() -> None:
    editor = Editor(EditState())
    editor.call("retouch_portrait", {"whiten_teeth": 0})
    assert [layer.name for layer in editor.state.layers] == ["Smooth skin", "Brighten eyes"]
    with pytest.raises(ToolError, match="Invalid retouch"):
        editor.call("retouch_portrait", {"smooth_skin": 200})
    with pytest.raises(ToolError, match="set to 0"):
        editor.call(
            "retouch_portrait",
            {"smooth_skin": 0, "remove_blemishes": False, "brighten_eyes": 0, "whiten_teeth": 0},
        )


def test_retouch_endpoint_is_one_history_step(
    client: TestClient, upload: Upload, settings: Settings
) -> None:
    doc = upload("portrait.jpg")
    res = client.post(f"/api/documents/{doc['id']}/retouch")
    assert res.status_code == 200
    body = res.json()
    assert [layer["name"] for layer in body["state"]["layers"]] == [
        "Smooth skin",
        "Brighten eyes",
        "Whiten teeth",
    ]
    assert body["undo_label"] == "Retouch portrait"
    loaded = get_store(settings).image(doc["id"])
    plain = render_state(loaded.proxy, EditState(), loaded.proxy_context)
    state = EditState.model_validate(body["state"])
    out = render_state(loaded.proxy, state, loaded.proxy_context)
    changed = np.abs(out - plain).max(axis=2) > 0.01
    assert changed[100:180, 180:280].any()  # the face changed
    assert not changed[:60, :60].any()  # the backdrop did not
