import numpy as np
import pytest

from photo_agent import operations as ops
from photo_agent.layers import EditState, Layer
from photo_agent.render import RenderContext, blend, luma, render, render_state

CTX = RenderContext()


@pytest.fixture
def gradient() -> np.ndarray:
    """A small image with a range of tones and colors."""
    ramp = np.linspace(0.05, 0.95, 32, dtype=np.float32)
    img = np.stack(np.meshgrid(ramp, ramp[::-1]), axis=-1)
    return np.concatenate([img, img[..., :1] * 0.5], axis=-1).astype(np.float32)


def state(*layers: Layer, framing: list[ops.OpBase] | None = None) -> EditState:
    return EditState.model_validate({"framing": framing or [], "layers": list(layers)})


def test_one_full_layer_matches_the_flat_render(gradient: np.ndarray) -> None:
    looks = [ops.Exposure(stops=0.5), ops.Saturation(amount=30)]
    layered = render_state(gradient, state(Layer(id="x", name="a", operations=looks)), CTX)
    np.testing.assert_allclose(layered, render(gradient, looks, CTX), atol=1e-6)


def test_hidden_and_transparent_layers_change_nothing(gradient: np.ndarray) -> None:
    looks = [ops.Exposure(stops=1)]
    hidden = Layer(id="x", name="a", operations=looks, visible=False)
    clear = Layer(id="x", name="b", operations=looks, opacity=0)
    np.testing.assert_allclose(render_state(gradient, state(hidden, clear), CTX), gradient)


def test_opacity_fades_a_layer(gradient: np.ndarray) -> None:
    looks = [ops.Exposure(stops=1)]
    full = render_state(gradient, state(Layer(id="x", name="a", operations=looks)), CTX)
    half = render_state(gradient, state(Layer(id="x", name="a", operations=looks, opacity=50)), CTX)
    unclipped = full < 0.999
    np.testing.assert_allclose(half[unclipped], ((gradient + full) / 2)[unclipped], atol=1e-5)


def test_layers_apply_bottom_to_top(gradient: np.ndarray) -> None:
    bottom = Layer(id="x", name="dark", operations=[ops.Exposure(stops=-1)])
    top = Layer(id="x", name="bw", operations=[ops.Saturation(amount=-100)])
    out = render_state(gradient, state(bottom, top), CTX)
    expected = render(gradient, [ops.Exposure(stops=-1), ops.Saturation(amount=-100)], CTX)
    np.testing.assert_allclose(out, expected, atol=1e-6)


def test_framing_applies_before_layers(gradient: np.ndarray) -> None:
    layer = Layer(id="x", name="v", operations=[ops.Vignette(amount=-50)])
    out = render_state(gradient, state(layer, framing=[ops.Crop(right=0.5)]), CTX)
    assert out.shape == (32, 16, 3)
    # The vignette is centered on the cropped frame, so its left and right edges match.
    np.testing.assert_allclose(out[:, 0] / gradient[:, 0], out[:, -1] / gradient[:, 15], atol=0.02)


def test_luminosity_blend_keeps_colors(gradient: np.ndarray) -> None:
    bright = render(gradient, [ops.Exposure(stops=1)], CTX)
    out = blend(gradient, bright, "luminosity")
    np.testing.assert_allclose(luma(out), luma(bright), atol=1e-5)
    np.testing.assert_allclose(
        out - luma(out)[..., None], gradient - luma(gradient)[..., None], atol=1e-5
    )


def test_color_blend_keeps_brightness(gradient: np.ndarray) -> None:
    gray = render(gradient, [ops.Saturation(amount=-100)], CTX)
    out = blend(gradient, gray, "color")
    np.testing.assert_allclose(luma(out), luma(gradient), atol=1e-5)


@pytest.mark.parametrize(
    ("mode", "darker"), [("multiply", True), ("screen", False), ("overlay", None)]
)
def test_classic_blend_modes(gradient: np.ndarray, mode: str, darker: bool | None) -> None:
    out = blend(gradient, gradient, mode)  # type: ignore[arg-type]
    if darker is True:
        assert (out <= gradient + 1e-6).all()
    elif darker is False:
        assert (out >= gradient - 1e-6).all()
    else:
        assert out.std() > gradient.std()


def test_layers_hold_only_looks() -> None:
    with pytest.raises(ValueError):
        crop = {"op": "crop", "aspect": "1:1"}
        Layer.model_validate({"id": "x", "name": "x", "operations": [crop]})
    with pytest.raises(ValueError):
        EditState.model_validate({"framing": [{"op": "exposure", "stops": 1}]})
