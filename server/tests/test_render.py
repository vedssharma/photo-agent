"""Render engine tests: each operation moves pixels the way its name promises, plus
golden-image comparisons so the look of an operation never changes by accident.

Regenerate the golden images after an intentional change with:
    UPDATE_GOLDEN=1 uv run pytest tests/test_render.py
"""

import os
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from photo_agent import imaging
from photo_agent import operations as ops
from photo_agent.layers import EditState
from photo_agent.render import (
    RenderCache,
    RenderContext,
    crop_box,
    largest_rotated_rect,
    luma,
    monotone_curve,
    render,
)

GOLDEN = Path(__file__).parent / "golden"
CTX = RenderContext()


@pytest.fixture(scope="module")
def landscape() -> imaging.Array:
    path = Path(__file__).parents[2] / "fixtures" / "photos" / "landscape.png"
    return imaging.resize_long_edge(imaging.decode(path.read_bytes()).pixels, 160)


def ramp(w: int = 64, h: int = 16) -> imaging.Array:
    """A horizontal gray ramp from black to white."""
    row = np.linspace(0.0, 1.0, w, dtype=np.float32)
    return np.repeat(np.repeat(row[None, :, None], h, axis=0), 3, axis=2)


def colorful() -> imaging.Array:
    rng = np.random.default_rng(0)
    return rng.uniform(0.15, 0.85, (32, 32, 3)).astype(np.float32)


def sat(x: imaging.Array) -> float:
    return float((x.max(axis=2) - x.min(axis=2)).mean())


def mean_luma(x: imaging.Array) -> float:
    return float(luma(x).mean())


def test_no_operations_is_identity(landscape: imaging.Array) -> None:
    np.testing.assert_allclose(render(landscape, [], CTX), landscape)


def test_render_does_not_modify_its_input(landscape: imaging.Array) -> None:
    before = landscape.copy()
    render(landscape, [ops.Exposure(stops=1), ops.Crop(right=0.5)], CTX)
    np.testing.assert_array_equal(landscape, before)


@pytest.mark.parametrize(
    ("op", "brighter"),
    [
        (ops.Exposure(stops=1), True),
        (ops.Exposure(stops=-1), False),
        (ops.Shadows(amount=60), True),
        (ops.Highlights(amount=-60), False),
        (ops.Whites(amount=50), True),
        (ops.Blacks(amount=-50), False),
    ],
)
def test_light_operations_change_brightness(op: ops.OpBase, brighter: bool) -> None:
    before, after = mean_luma(ramp()), mean_luma(render(ramp(), [op], CTX))
    assert (after > before) == brighter


def test_shadows_leave_highlights_and_highlights_leave_shadows() -> None:
    x = ramp()
    lifted = render(x, [ops.Shadows(amount=80)], CTX)
    assert np.abs(lifted[:, -4:] - x[:, -4:]).max() < 0.01
    recovered = render(x, [ops.Highlights(amount=-80)], CTX)
    assert np.abs(recovered[:, :8] - x[:, :8]).max() < 0.01


def test_contrast_spreads_or_compresses_tones() -> None:
    x = ramp()
    assert float(render(x, [ops.Contrast(amount=60)], CTX).std()) > float(x.std())
    assert float(render(x, [ops.Contrast(amount=-60)], CTX).std()) < float(x.std())


def test_saturation_and_vibrance() -> None:
    x = colorful()
    assert sat(render(x, [ops.Saturation(amount=50)], CTX)) > sat(x)
    gray = render(x, [ops.Saturation(amount=-100)], CTX)
    assert sat(gray) < 1e-5
    assert sat(render(x, [ops.Vibrance(amount=50)], CTX)) > sat(x)


def test_white_balance_warms_and_cools() -> None:
    x = np.full((8, 8, 3), 0.5, np.float32)
    warm = render(x, [ops.WhiteBalance(temperature=50)], CTX)
    assert warm[..., 0].mean() > warm[..., 2].mean()
    cool = render(x, [ops.WhiteBalance(temperature=-50)], CTX)
    assert cool[..., 2].mean() > cool[..., 0].mean()
    magenta = render(x, [ops.WhiteBalance(tint=50)], CTX)
    assert magenta[..., 1].mean() < magenta[..., 0].mean()


def test_hsl_only_touches_its_band() -> None:
    blue = np.zeros((4, 8, 3), np.float32)
    blue[:, :4] = [0.2, 0.3, 0.8]  # blue
    blue[:, 4:] = [0.8, 0.2, 0.2]  # red
    out = render(blue, [ops.HSL(band="blue", saturation=-100)], CTX)
    assert sat(out[:, :4]) < 0.05
    np.testing.assert_allclose(out[:, 4:], blue[:, 4:], atol=1e-3)


def test_detail_operations(landscape: imaging.Array) -> None:
    def edges(x: imaging.Array) -> float:
        return float(np.abs(np.diff(luma(x), axis=1)).mean())

    assert edges(render(landscape, [ops.Sharpen(amount=80)], CTX)) > edges(landscape)
    assert edges(render(landscape, [ops.Clarity(amount=80)], CTX)) > edges(landscape)
    noisy = landscape + np.random.default_rng(1).normal(0, 0.05, landscape.shape).astype(np.float32)
    denoised = render(noisy, [ops.NoiseReduction(luminance=80, color=80)], CTX)
    assert edges(denoised) < edges(np.clip(noisy, 0, 1))


def test_dehaze_adds_and_removes_contrast(landscape: imaging.Array) -> None:
    hazy = landscape * 0.6 + 0.35
    assert float(render(hazy, [ops.Dehaze(amount=80)], CTX).std()) > float(hazy.std())
    assert float(render(landscape, [ops.Dehaze(amount=-80)], CTX).std()) < float(landscape.std())


def test_crop_with_aspect_ratio() -> None:
    assert crop_box(ops.Crop(aspect="4:5"), 1000, 1000, 1.0) == (100, 0, 900, 1000)
    assert crop_box(ops.Crop(aspect="1:1"), 400, 300, 4 / 3) == (50, 0, 350, 300)
    assert crop_box(ops.Crop(left=0.5, top=0.5), 400, 300, 4 / 3) == (200, 150, 400, 300)
    assert crop_box(ops.Crop(aspect="original"), 300, 300, 3 / 2) == (0, 50, 300, 250)
    out = render(np.zeros((300, 400, 3), np.float32), [ops.Crop(aspect="16:9")], CTX)
    assert out.shape == (225, 400, 3)


def test_rotate_flip_and_straighten() -> None:
    x = np.zeros((20, 40, 3), np.float32)
    x[0, 0] = 1.0  # top-left marker
    rotated = render(x, [ops.Rotate(degrees=90)], CTX)
    assert rotated.shape == (40, 20, 3)
    assert rotated[0, -1, 0] == 1.0  # top-left moves to top-right on a clockwise turn
    flipped = render(x, [ops.Flip(axis="horizontal")], CTX)
    assert flipped[0, -1, 0] == 1.0
    leveled = render(np.ones((100, 200, 3), np.float32), [ops.Straighten(angle=5)], CTX)
    assert leveled.shape[0] < 100 and leveled.shape[1] < 200
    assert leveled.min() > 0.99  # no empty corners left after the auto-crop


def test_largest_rotated_rect_is_identity_at_zero() -> None:
    assert largest_rotated_rect(200, 100, 0.0) == pytest.approx((200, 100))


def test_vignette_darkens_corners_not_center() -> None:
    x = np.full((50, 50, 3), 0.6, np.float32)
    out = render(x, [ops.Vignette(amount=-80)], CTX)
    assert out[0, 0, 0] < 0.4
    assert out[25, 25, 0] == pytest.approx(0.6, abs=1e-3)


def test_grain_is_deterministic_per_operation(landscape: imaging.Array) -> None:
    op = ops.Grain(amount=50)
    a, b = render(landscape, [op], CTX), render(landscape, [op], CTX)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, landscape)


def test_tone_curve_is_monotone_and_hits_its_points() -> None:
    curve = monotone_curve([(0, 0.1), (0.25, 0.2), (0.75, 0.85), (1, 1)], samples=101)
    assert curve[0] == pytest.approx(0.1)
    assert curve[25] == pytest.approx(0.2, abs=1e-3)
    assert np.all(np.diff(curve) >= -1e-6)
    out = render(ramp(), [ops.ToneCurve(points=[(0, 0.2), (1, 1)], channel="blue")], CTX)
    assert out[0, 0, 2] == pytest.approx(0.2)
    assert out[0, 0, 0] == 0.0


def test_preview_and_full_resolution_renders_agree(landscape: imaging.Array) -> None:
    """A downscaled render of the full-res result should match rendering the proxy."""
    full = imaging.resize_long_edge(
        imaging.decode(
            (Path(__file__).parents[2] / "fixtures" / "photos" / "landscape.png").read_bytes()
        ).pixels,
        640,
    )
    proxy = imaging.resize_long_edge(full, 320)
    graph = [
        ops.Exposure(stops=0.3),
        ops.Clarity(amount=40),
        ops.Sharpen(amount=40),
        ops.Crop(aspect="4:5"),
        ops.Vignette(amount=-40),
    ]
    big = render(full, graph, RenderContext(scale=1.0, source_aspect=640 / 427))
    small = render(proxy, graph, RenderContext(scale=0.5, source_aspect=640 / 427))
    assert abs(big.shape[0] / 2 - small.shape[0]) <= 1
    assert abs(big.shape[1] / 2 - small.shape[1]) <= 1
    down = cv2.resize(big, small.shape[1::-1], interpolation=cv2.INTER_AREA)
    assert float(np.abs(down - small).mean()) < 0.02


def test_render_cache_reuses_results(landscape: imaging.Array) -> None:
    cache = RenderCache(size=2)
    state = EditState.from_operations([ops.Exposure(stops=1, id="a")], "x")
    first = cache.get_or_render("doc", landscape, state, CTX)
    assert cache.get_or_render("doc", landscape, state, CTX) is first
    state.layers[0].operations[0] = ops.Exposure(stops=2, id="a")
    other = cache.get_or_render("doc", landscape, state, CTX)
    assert other is not first


# Golden images: one representative setting per operation, rendered on the landscape fixture.
GOLDEN_CASES: dict[str, list[ops.OpBase]] = {
    "exposure": [ops.Exposure(stops=0.7)],
    "contrast": [ops.Contrast(amount=50)],
    "highlights": [ops.Highlights(amount=-70)],
    "shadows": [ops.Shadows(amount=70)],
    "whites": [ops.Whites(amount=40)],
    "blacks": [ops.Blacks(amount=-40)],
    "white_balance": [ops.WhiteBalance(temperature=40, tint=10)],
    "vibrance": [ops.Vibrance(amount=60)],
    "saturation": [ops.Saturation(amount=-60)],
    "hsl": [ops.HSL(band="blue", hue=-20, saturation=40, luminance=-30)],
    "sharpen": [ops.Sharpen(amount=100, radius=1.5)],
    "noise_reduction": [ops.NoiseReduction(luminance=70, color=70)],
    "clarity": [ops.Clarity(amount=60)],
    "dehaze": [ops.Dehaze(amount=60)],
    "crop": [ops.Crop(left=0.1, top=0.1, right=0.9, bottom=0.9, aspect="1:1")],
    "rotate": [ops.Rotate(degrees=270)],
    "straighten": [ops.Straighten(angle=-4)],
    "flip": [ops.Flip(axis="vertical")],
    "vignette": [ops.Vignette(amount=-60, midpoint=40)],
    "grain": [ops.Grain(amount=60, size=50, id="golden01")],
    "tone_curve": [ops.ToneCurve(points=[(0, 0.05), (0.3, 0.22), (0.7, 0.8), (1, 0.97)])],
}


MODEL_OPERATIONS = {"remove"}
"""Operations whose result comes from an AI model (tested in their own modules)."""


def test_golden_cases_cover_every_operation() -> None:
    covered = {op.op for case in GOLDEN_CASES.values() for op in case}  # type: ignore[attr-defined]
    assert covered == set(ops.OPERATIONS_BY_NAME) - MODEL_OPERATIONS


@pytest.mark.parametrize("name", sorted(GOLDEN_CASES))
def test_golden_image(name: str, landscape: imaging.Array) -> None:
    out = imaging.to_uint8(render(landscape, GOLDEN_CASES[name], CTX))
    path = GOLDEN / f"{name}.png"
    if os.environ.get("UPDATE_GOLDEN") or not path.exists():
        if not os.environ.get("UPDATE_GOLDEN"):
            pytest.fail(f"missing golden image {path.name}; run with UPDATE_GOLDEN=1")
        Image.fromarray(out).save(path, optimize=True)
        return
    expected = np.asarray(Image.open(path).convert("RGB"))
    assert out.shape == expected.shape
    diff = np.abs(out.astype(np.int16) - expected.astype(np.int16))
    # Tolerate tiny differences between OpenCV/NumPy builds, not visible changes.
    assert float(diff.mean()) < 0.5, f"{name} looks different (mean diff {diff.mean():.2f})"
    assert int(diff.max()) <= 8
