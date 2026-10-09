"""Phase 3's goal, end to end: "remove the person on the left and make the sky bluer",
then refine the removal by brushing on its mask."""

import asyncio
from typing import Any

import cv2
import numpy as np
from fake_model import FakeModel, text, tool
from fastapi.testclient import TestClient

from photo_agent import imaging
from photo_agent.agent import AgentEvent, AgentService
from photo_agent.layers import EditState
from photo_agent.render import RenderCache, render_state
from photo_agent.routes import get_store
from photo_agent.settings import Settings

W, H = 600, 400
LEFT = (100, 150)  # the left person's body spans these columns
RIGHT = (450, 500)


def street(people: bool = True) -> imaging.Array:
    """Blue sky over a field, with two people standing in it."""
    rng = np.random.default_rng(3)
    img = np.zeros((H, W, 3), np.float32)
    t = np.linspace(0, 1, H // 2, dtype=np.float32)[:, None, None]
    img[: H // 2] = (1 - t) * np.array([0.45, 0.6, 0.85]) + t * np.array([0.7, 0.78, 0.9])
    ground = np.array([0.38, 0.42, 0.26], np.float32)
    img[H // 2 :] = ground + rng.normal(0, 0.02, (H - H // 2, W, 1)).astype(np.float32)
    if people:
        for x0, x1 in (LEFT, RIGHT):
            cv2.rectangle(img, (x0, 170), (x1, 340), (0.15, 0.15, 0.3), -1)
            cv2.circle(img, ((x0 + x1) // 2, 145), 22, (0.85, 0.65, 0.55), -1)
    return np.clip(img, 0, 1)


def blueness(a: imaging.Array) -> float:
    return float((a[..., 2] - a[..., 0]).mean())


def test_remove_the_person_on_the_left_and_make_the_sky_bluer(
    client: TestClient, settings: Settings
) -> None:
    res = client.post(
        "/api/documents", files={"file": ("street.png", imaging.encode_png(street()))}
    )
    doc_id = res.json()["id"]
    left_box = [LEFT[0] / W - 0.03, 110 / H, LEFT[1] / W + 0.03, 350 / H]
    model = FakeModel(
        [
            tool(
                "add_layer",
                {
                    "name": "Remove the person on the left",
                    "mask": {
                        "kind": "semantic",
                        "target": "object",
                        "box": left_box,
                        "description": "the person on the left",
                    },
                },
            ),
            tool("remove", {"grow": 30}, id="toolu_remove"),
            tool(
                "add_layer",
                {"name": "Bluer sky", "mask": {"kind": "semantic", "target": "sky"}},
            ),
            tool("hsl", {"band": "blue", "saturation": 50, "luminance": -10}),
        ],
        [text("I removed the person on the left and deepened the blue of the sky.")],
    )
    events: list[AgentEvent] = []

    async def emit(event: AgentEvent) -> None:
        events.append(event)

    store = get_store(settings)
    service = AgentService(store, RenderCache(), model)
    doc = asyncio.run(
        service.run_turn(doc_id, "remove the person on the left and make the sky bluer", emit)
    )
    assert [layer.name for layer in doc.state.layers] == [
        "Remove the person on the left",
        "Bluer sky",
    ]

    loaded = store.image(doc_id)
    ctx = loaded.proxy_context
    before = render_state(loaded.proxy, EditState(), ctx)
    after = render_state(loaded.proxy, doc.state, ctx)
    clean = street(people=False)

    # The person on the left is gone: their spot now looks like the field and sky behind.
    body = (slice(200, 330), slice(*LEFT))
    assert np.abs(after[body] - clean[body]).mean() < 0.06
    assert np.abs(before[body] - clean[body]).mean() > 0.15
    # The person on the right is untouched.
    right = (slice(200, 330), slice(*RIGHT))
    np.testing.assert_allclose(after[right], before[right], atol=0.02)
    # The sky is bluer; the field is not.
    sky = (slice(5, 60), slice(250, 400))
    assert blueness(after[sky]) > blueness(before[sky]) + 0.05
    field = (slice(360, 395), slice(250, 400))
    np.testing.assert_allclose(after[field], before[field], atol=0.02)

    # The person refines the removal by brushing. Painting more onto the selection
    # removes more (a shadow, a bag the model missed); erasing brings things back.
    def refine(*strokes: dict[str, Any]) -> imaging.Array:
        state = doc.state.model_dump(mode="json")
        state["layers"][0]["mask"]["strokes"] = list(strokes)
        res = client.post(
            f"/api/documents/{doc_id}/edits",
            json={"label": "Touch up the selection", "state": state},
        )
        assert res.status_code == 200, res.text
        return render_state(loaded.proxy, EditState.model_validate(res.json()["state"]), ctx)

    bag = (slice(300, 340), slice(LEFT[1] + 30, LEFT[1] + 60))
    painted = refine({"points": [[(LEFT[1] + 45) / W, 320 / H]], "size": 0.04, "hardness": 100})
    assert np.abs(painted[bag] - before[bag]).max() > 0.02  # filled in again
    assert np.abs(painted[body] - clean[body]).mean() < 0.06

    head = [(LEFT[0] + LEFT[1]) / 2 / W, 145 / H]
    erased = refine({"points": [head], "size": 0.06, "hardness": 100, "erase": True})
    face = (slice(135, 155), slice(LEFT[0] + 15, LEFT[1] - 15))
    np.testing.assert_allclose(erased[face], before[face], atol=0.03)
    assert np.abs(erased[body] - clean[body]).mean() < 0.06
    head_gone = (slice(130, 160), slice(LEFT[0] + 10, LEFT[1] - 10))
    assert np.abs(after[head_gone] - clean[head_gone]).mean() < 0.06
