"""Variants: several takes on a generative edit, to compare and pick from."""

import asyncio
import io
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from fake_model import FakeModel, text, tool
from fastapi.testclient import TestClient
from PIL import Image

from photo_agent import operations as ops
from photo_agent import variants
from photo_agent.agent import AgentEvent, AgentService, Editor, ToolError
from photo_agent.generative import cache_identity
from photo_agent.layers import EditState, Layer
from photo_agent.masks import RadialGradientMask
from photo_agent.render import RenderCache
from photo_agent.routes import get_store
from photo_agent.settings import Settings

Upload = Callable[[str], dict[str, Any]]

SPOT = RadialGradientMask(center=[0.3, 0.5], radius_x=0.08, radius_y=0.1, feather=0)


def filled(seed: int = 7) -> EditState:
    op = ops.Generate(id="gen", prompt="a red kite", seed=seed)
    return EditState(layers=[Layer(id="Lk", name="Kite", mask=SPOT, operations=[op])])


def test_offer_keeps_the_current_take_first() -> None:
    state = variants.offer(filled(), "gen", 3)
    op = variants.generative_op(state, "gen")
    assert op.options[0] == 7 and len(set(op.options)) == 3
    assert op.seed == 7
    picked = variants.with_seed(state, "gen", op.options[2])
    assert variants.generative_op(picked, "gen").seed == op.options[2]
    # What is on offer does not change what renders.
    assert cache_identity(op) == cache_identity(variants.generative_op(filled(), "gen"))
    with pytest.raises(ValueError):
        variants.offer(filled(), "gen", 9)
    with pytest.raises(variants.NotGenerativeError):
        variants.offer(
            EditState(layers=[Layer(id="L", name="x", operations=[ops.Exposure(id="e", stops=1)])]),
            "e",
        )


def test_contact_sheet_lines_takes_up() -> None:
    takes = [np.full((40, 60, 3), v, np.float32) for v in (0.2, 0.5, 0.8)]
    sheet = variants.contact_sheet(takes, gap=8)
    assert sheet.shape == (40, 3 * 60 + 2 * 8, 3)


def test_options_api(client: TestClient, settings: Settings, upload: Upload) -> None:
    doc = upload("landscape.png")
    store = get_store(settings)
    d = store.get(doc["id"])
    d.edit_by_hand("Kite", filled())
    store.save(d)

    res = client.post(f"/api/documents/{doc['id']}/operations/gen/options", json={"count": 3})
    assert res.status_code == 200
    body = res.json()
    seeds = body["state"]["layers"][0]["operations"][0]["options"]
    assert len(seeds) == 3 and body["history"][-1]["label"].startswith("Options for Generate")
    loaded = store.image(doc["id"])
    assert loaded.vision is not None
    assert len(loaded.vision.worker.jobs(doc["id"])) == 3  # each take generated once

    shots = []
    for seed in seeds:
        img = client.get(f"/api/documents/{doc['id']}/operations/gen/options/{seed}")
        assert img.status_code == 200 and img.headers["content-type"] == "image/jpeg"
        shots.append(np.asarray(Image.open(io.BytesIO(img.content)), np.float32))
    assert np.abs(shots[0] - shots[1]).max() > 2  # different takes
    assert len(loaded.vision.worker.jobs(doc["id"])) == 3  # nothing generated again
    assert client.get(f"/api/documents/{doc['id']}/operations/gen/options/12345").status_code == 404
    assert client.post(f"/api/documents/{doc['id']}/operations/nope/options").status_code == 404


def test_the_agent_sees_the_takes_side_by_side(upload: Upload, settings: Settings) -> None:
    doc = upload("landscape.png")
    store = get_store(settings)
    d = store.get(doc["id"])
    d.edit_by_hand("Kite", filled())
    store.save(d)
    model = FakeModel(
        [tool("show_options", {"id": "gen", "count": 2})],
        [text("Take 2 has a bigger kite.")],
    )

    async def emit(event: AgentEvent) -> None:
        pass

    service = AgentService(store, RenderCache(), model)
    result = asyncio.run(service.run_turn(doc["id"], "show me 2 options", emit))
    op = variants.generative_op(result.state, "gen")
    assert len(op.options) == 2 and op.seed == 7
    content = model.calls[1][-1]["content"][0]["content"]
    assert content[0]["type"] == "text" and "take 1 is current" in content[0]["text"]
    assert content[1]["type"] == "image"  # the contact sheet


def test_show_options_needs_a_generative_operation() -> None:
    with pytest.raises(ToolError):
        Editor(EditState()).call("show_options", {"id": "nope"})
    with pytest.raises(ToolError):
        Editor(filled()).call("show_options", {"id": "gen", "count": 7})
