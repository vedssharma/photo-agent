import io
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from helpers import step
from PIL import Image

from photo_agent import operations as ops
from photo_agent.layers import EditState, Layer
from photo_agent.masks import (
    BrushMask,
    BrushStroke,
    LinearGradientMask,
    LuminosityMask,
    Mask,
    RadialGradientMask,
    SemanticMask,
    describe_mask,
    render_mask,
)
from photo_agent.render import RenderContext, render_state
from photo_agent.routes import get_store
from photo_agent.settings import Settings

Upload = Callable[[str], dict[str, Any]]
GRAY = np.full((60, 80, 3), 0.5, np.float32)


def test_linear_gradient_fades_from_start_to_end() -> None:
    alpha = render_mask(LinearGradientMask(start=[0.5, 0], end=[0.5, 0.5]), GRAY)
    assert alpha[0].min() > 0.99
    assert alpha[30:].max() < 0.01
    assert alpha[15].mean() == pytest.approx(0.5, abs=0.05)
    inverted = render_mask(LinearGradientMask(start=[0.5, 0], end=[0.5, 0.5], invert=True), GRAY)
    np.testing.assert_allclose(inverted, 1 - alpha)


def test_radial_gradient_is_strongest_in_the_middle() -> None:
    mask = RadialGradientMask(center=[0.5, 0.5], radius_x=0.25, radius_y=0.25, feather=50)
    alpha = render_mask(mask, GRAY)
    assert alpha[30, 40] == pytest.approx(1.0)
    assert alpha[0, 0] == 0.0
    # The ellipse is relative to each side, so it reaches 0.25 of the width sideways.
    assert alpha[30, 40 + 21] == 0.0
    hard = render_mask(mask.model_copy(update={"feather": 0}), GRAY)
    assert set(np.unique(hard)) == {0.0, 1.0}


def test_luminosity_range_selects_tones() -> None:
    ramp = np.repeat(np.linspace(0, 1, 101, dtype=np.float32)[None, :, None], 3, axis=2)
    alpha = render_mask(LuminosityMask(low=0.6, high=1, feather=0.1), ramp)[0]
    assert alpha[:50].max() == 0.0
    assert alpha[60:].min() == pytest.approx(1.0)
    shadows = render_mask(LuminosityMask(low=0, high=0.3, feather=0.05), ramp)[0]
    assert shadows[0] == 1.0 and shadows[40] == 0.0


def test_brush_paints_and_erases() -> None:
    stroke = BrushStroke(points=[[0.1, 0.5], [0.9, 0.5]], size=0.05, hardness=100)
    alpha = render_mask(BrushMask(strokes=[stroke]), GRAY)
    assert alpha[30, 40] == pytest.approx(1.0)
    assert alpha[5, 40] == 0.0
    dot = BrushStroke(points=[[0.5, 0.5]], size=0.1, hardness=0, erase=True)
    erased = render_mask(BrushMask(strokes=[stroke, dot]), GRAY)
    assert erased[30, 40] < 0.05
    assert erased[30, 12] == pytest.approx(1.0)


def test_brush_mask_looks_the_same_at_any_resolution() -> None:
    mask = BrushMask(strokes=[BrushStroke(points=[[0.2, 0.2], [0.8, 0.7]], size=0.04)])
    small = render_mask(mask, np.zeros((150, 200, 3), np.float32))
    big = render_mask(mask, np.zeros((1500, 2000, 3), np.float32))
    assert float(np.abs(big[::10, ::10] - small).mean()) < 0.02


def test_masked_layer_applies_only_inside_the_mask() -> None:
    layer = Layer(
        id="L",
        name="sky",
        operations=[ops.Exposure(stops=-1)],
        mask=LinearGradientMask(start=[0.5, 0], end=[0.5, 0.5]),
    )
    out = render_state(GRAY, EditState(layers=[layer]), RenderContext())
    assert out[0].mean() < 0.4
    np.testing.assert_allclose(out[45:], GRAY[45:], atol=1e-6)


@pytest.mark.parametrize(
    ("mask", "text"),
    [
        (BrushMask(strokes=[]), "brush mask (0 strokes)"),
        (LuminosityMask(low=0.6, invert=True), "inverted luminosity mask 0.60-1.00"),
    ],
)
def test_masks_describe_themselves(mask: Mask, text: str) -> None:
    assert describe_mask(mask) == text


def test_layer_mask_endpoint(client: TestClient, upload: Upload, settings: Settings) -> None:
    doc = upload("landscape.png")
    store = get_store(settings)
    stored = store.get(doc["id"])
    sky = step("sky", ops.Exposure(stops=-0.5))
    sky.state.layers[0].mask = LinearGradientMask(start=[0.5, 0], end=[0.5, 0.5])
    stored.commit(sky)
    store.save(stored)
    layer_id = sky.state.layers[0].id

    res = client.get(f"/api/documents/{doc['id']}/layers/{layer_id}/mask")
    assert res.status_code == 200
    assert res.headers["content-type"] == "image/png"
    img = np.asarray(Image.open(io.BytesIO(res.content)))
    assert img.ndim == 2
    assert img[0].min() > 250 and img[-1].max() < 5
    assert client.get(f"/api/documents/{doc['id']}/layers/nope/mask").status_code == 404


def test_brush_touch_ups_refine_a_semantic_mask() -> None:
    x = np.zeros((100, 200, 3), np.float32)

    def left_half(mask: SemanticMask, shape: tuple[int, int]) -> np.ndarray:
        alpha = np.zeros(shape, np.float32)
        alpha[:, : shape[1] // 2] = 1
        return alpha

    plain = SemanticMask(target="sky")
    assert render_mask(plain, x, left_half)[50, 150] == 0
    touched = SemanticMask(
        target="sky",
        strokes=[
            BrushStroke(points=[[0.75, 0.5]], size=0.05, hardness=100),
            BrushStroke(points=[[0.25, 0.5]], size=0.05, hardness=100, erase=True),
        ],
    )
    alpha = render_mask(touched, x, left_half)
    assert alpha[50, 150] == pytest.approx(1, abs=0.01)  # painted on
    assert alpha[50, 50] == pytest.approx(0, abs=0.01)  # erased
    assert alpha[10, 20] == 1 and alpha[10, 180] == 0  # the rest as the model found it
    assert "touched up with 2 brush strokes" in describe_mask(touched)
    inverted = touched.model_copy(update={"invert": True})
    np.testing.assert_allclose(render_mask(inverted, x, left_half), 1 - alpha)


def test_the_agent_keeps_touch_ups_when_adjusting_a_selection() -> None:
    from photo_agent.agent import Editor

    stroke = BrushStroke(points=[[0.2, 0.2]], size=0.05)
    mask = SemanticMask(target="object", box=[0.1, 0.1, 0.4, 0.9], strokes=[stroke])
    editor = Editor(EditState(layers=[Layer(id="L1", name="Dog", mask=mask)]))
    moved = {"kind": "semantic", "target": "object", "box": [0.1, 0.1, 0.5, 0.9]}
    editor.call("update_layer", {"id": "L1", "changes": {"mask": moved}})
    assert editor.state.layers[0].mask.strokes == [stroke]  # type: ignore[union-attr]
    sky = {"kind": "semantic", "target": "sky"}
    editor.call("update_layer", {"id": "L1", "changes": {"mask": sky}})
    assert editor.state.layers[0].mask.strokes == []  # type: ignore[union-attr]
