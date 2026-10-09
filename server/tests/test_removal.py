"""Object removal: removal layers fill in what their mask selects."""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from photo_agent import operations as ops
from photo_agent.agent import Editor, ToolError
from photo_agent.layers import EditState, Layer
from photo_agent.masks import RadialGradientMask, SemanticMask
from photo_agent.render import render_state
from photo_agent.routes import get_store
from photo_agent.settings import Settings
from photo_agent.vision import classical

Upload = Callable[[str], dict[str, Any]]


def test_classical_inpaint_fills_from_the_surroundings() -> None:
    h, w = 120, 160
    ramp = np.linspace(0.2, 0.8, w, dtype=np.float32)
    image = np.repeat(np.repeat(ramp[None, :, None], h, 0), 3, 2)
    spoiled = image.copy()
    hole = np.zeros((h, w), np.float32)
    hole[40:80, 60:100] = 1
    spoiled[40:80, 60:100] = [0.9, 0.1, 0.1]  # a red sign in the way
    filled = classical.inpaint(spoiled, hole)
    np.testing.assert_array_equal(filled[hole == 0], spoiled[hole == 0])
    assert np.abs(filled[40:80, 60:100] - image[40:80, 60:100]).mean() < 0.06


def removal(mask: Any, grow: float = 20, **layer: Any) -> Layer:
    return Layer(
        id="Lremove",
        name="Remove the spot",
        mask=mask,
        operations=[ops.Remove(grow=grow)],
        **layer,
    )


SPOT = RadialGradientMask(center=[0.3, 0.5], radius_x=0.06, radius_y=0.09, feather=0)


def test_removal_changes_only_the_selection(settings: Settings, upload: Upload) -> None:
    doc = upload("landscape.png")
    loaded = get_store(settings).image(doc["id"])
    before = render_state(loaded.proxy, EditState(), loaded.proxy_context)
    after = render_state(loaded.proxy, EditState(layers=[removal(SPOT)]), loaded.proxy_context)
    h, w = before.shape[:2]
    changed = np.abs(after - before).max(axis=2) > 0.02
    assert changed[round(h * 0.5), round(w * 0.3)]
    assert not changed[:, round(w * 0.5) :].any()
    assert not changed[: round(h * 0.3)].any()
    # Hidden, a removal does nothing; rendered again, it reuses the fill.
    hidden = EditState(layers=[removal(SPOT, visible=False)])
    np.testing.assert_array_equal(render_state(loaded.proxy, hidden, loaded.proxy_context), before)
    vision = loaded.vision
    assert vision is not None
    jobs = len(vision.worker.jobs(doc["id"]))
    render_state(loaded.proxy, EditState(layers=[removal(SPOT)]), loaded.proxy_context)
    assert len(vision.worker.jobs(doc["id"])) == jobs


def test_removals_apply_before_adjustments(settings: Settings, upload: Upload) -> None:
    doc = upload("landscape.png")
    loaded = get_store(settings).image(doc["id"])
    brighter = Layer(id="Lb", name="Brighter", operations=[ops.Exposure(stops=0.5)])
    below = EditState(layers=[removal(SPOT), brighter])
    above = EditState(layers=[brighter, removal(SPOT)])
    np.testing.assert_allclose(
        render_state(loaded.proxy, below, loaded.proxy_context),
        render_state(loaded.proxy, above, loaded.proxy_context),
        atol=1e-6,
    )


def test_a_removal_stands_alone_in_its_layer() -> None:
    with pytest.raises(ValidationError, match="only operation"):
        Layer(id="L1", name="x", operations=[ops.Remove(), ops.Exposure(stops=1)])
    assert removal(SPOT).is_removal
    assert not Layer(id="L2", name="y").is_removal


def test_the_agent_removes_things_in_their_own_layer() -> None:
    editor = Editor(EditState())
    editor.call("add_layer", {"name": "Brighter"})
    editor.call("exposure", {"stops": 0.3})
    with pytest.raises(ToolError, match="own new layer"):
        editor.call("remove", {})
    person = {"kind": "semantic", "target": "object", "box": [0.1, 0.2, 0.3, 0.9]}
    editor.call("add_layer", {"name": "Remove the person on the left", "mask": person})
    editor.call("remove", {"grow": 30})
    with pytest.raises(ToolError, match="removes something"):
        editor.call("contrast", {"amount": 10})
    layer = editor.state.layers[-1]
    assert layer.is_removal
    assert isinstance(layer.mask, SemanticMask)


def test_removing_an_object_through_the_api(client: TestClient, upload: Upload) -> None:
    doc = upload("phone.heic")
    eye = SemanticMask(target="object", points=[{"x": 0.38, "y": 0.4}], description="the eye")
    state = EditState(layers=[removal(eye)]).model_dump(mode="json")
    res = client.post(
        f"/api/documents/{doc['id']}/edits", json={"label": "Remove the eye", "state": state}
    )
    assert res.status_code == 200, res.text
    tasks = [j["task"] for j in client.get(f"/api/documents/{doc['id']}/jobs").json()]
    assert tasks == ["segment_object", "inpaint"]
    bad = EditState(layers=[removal(eye)]).model_dump(mode="json")
    bad["layers"][0]["operations"].append({"op": "exposure", "stops": 1})
    res = client.post(f"/api/documents/{doc['id']}/edits", json={"label": "x", "state": bad})
    assert res.status_code == 422
