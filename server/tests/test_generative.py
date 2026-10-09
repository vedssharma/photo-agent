"""Generative fill: new content painted where a layer's mask selects, recorded so it
renders the same at any size."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from photo_agent import operations as ops
from photo_agent.agent import TOOLS, Editor, ToolError
from photo_agent.compositing import harmonize
from photo_agent.generative import job_params, stamp
from photo_agent.layers import EditState, Layer
from photo_agent.masks import RadialGradientMask
from photo_agent.recipes import portable
from photo_agent.render import (
    expand_box,
    extend_canvas,
    render_layer_mask,
    render_state,
    transfer_light,
)
from photo_agent.routes import get_store
from photo_agent.settings import Settings
from photo_agent.vision import classical
from photo_agent.vision.backends import Env, Progress, Task, WorkerConfig
from photo_agent.vision.tasks import TASKS
from photo_agent.vision.worker import PREFER, ModelWorker

Upload = Callable[[str], dict[str, Any]]

SPOT = RadialGradientMask(center=[0.3, 0.5], radius_x=0.06, radius_y=0.09, feather=0)


def fill(prompt: str = "a potted fern", seed: int | None = 7, **layer: Any) -> Layer:
    return Layer(
        id="Lgen",
        name="Add a plant",
        mask=layer.pop("mask", SPOT),
        operations=[ops.Generate(prompt=prompt, seed=seed)],
        **layer,
    )


def test_classical_fallback_fills_only_the_mask() -> None:
    image = np.full((60, 80, 3), 0.5, np.float32)
    image[20:40, 30:50] = [1, 0, 0]
    mask = np.zeros((60, 80), np.float32)
    mask[20:40, 30:50] = 1
    out = classical.generate(image, mask=mask, prompt="anything", seed=3)
    np.testing.assert_array_equal(out[mask == 0], image[mask == 0])
    assert np.abs(out[25:35, 35:45] - 0.5).mean() < 0.1
    assert classical.generate(image, prompt="no mask") is image


def test_generative_fill_changes_only_the_selection(settings: Settings, upload: Upload) -> None:
    doc = upload("landscape.png")
    loaded = get_store(settings).image(doc["id"])
    before = render_state(loaded.proxy, EditState(), loaded.proxy_context)
    after = render_state(loaded.proxy, EditState(layers=[fill()]), loaded.proxy_context)
    h, w = before.shape[:2]
    diff = np.abs(after - before).max(axis=2)
    inside = diff[round(h * 0.45) : round(h * 0.55), round(w * 0.27) : round(w * 0.33)]
    assert inside.mean() > 0.03
    assert not (diff[:, round(w * 0.5) :] > 0.02).any()
    jobs = loaded.vision.worker.jobs(doc["id"]) if loaded.vision else []
    assert [j.task for j in jobs] == ["generate"]


def test_a_result_is_generated_once_for_every_render_size(
    settings: Settings, upload: Upload
) -> None:
    doc = upload("landscape.png")
    loaded = get_store(settings).image(doc["id"])
    state = EditState(layers=[fill()])
    preview = render_state(loaded.proxy, state, loaded.proxy_context)
    full = render_state(loaded.source.pixels, state, loaded.full_context)
    assert loaded.vision is not None
    assert len(loaded.vision.worker.jobs(doc["id"])) == 1
    # The export shows the same take as the preview, at its own resolution.
    small = cv2.resize(full, preview.shape[1::-1], interpolation=cv2.INTER_AREA)
    assert np.abs(small - preview).mean() < 0.02
    # A new seed is a new take.
    render_state(loaded.proxy, EditState(layers=[fill(seed=8)]), loaded.proxy_context)
    assert len(loaded.vision.worker.jobs(doc["id"])) == 2


def test_generative_layers_render_before_adjustments(settings: Settings, upload: Upload) -> None:
    doc = upload("landscape.png")
    loaded = get_store(settings).image(doc["id"])
    brighter = Layer(id="Lb", name="Brighter", operations=[ops.Exposure(stops=0.5)])
    np.testing.assert_allclose(
        render_state(loaded.proxy, EditState(layers=[fill(), brighter]), loaded.proxy_context),
        render_state(loaded.proxy, EditState(layers=[brighter, fill()]), loaded.proxy_context),
        atol=1e-6,
    )


def test_stamp_records_a_seed_and_the_model() -> None:
    state = EditState(layers=[fill(seed=None)])
    stamped = stamp(state, lambda task: f"model-for-{task}")
    (op,) = stamped.layers[0].operations
    assert isinstance(op, ops.Generate)
    assert op.seed is not None and op.model == "model-for-generate"
    assert stamp(stamped, lambda task: "other") is stamped
    assert state.layers[0].operations[0].seed is None  # type: ignore[union-attr]
    assert job_params(op) == {"seed": op.seed, "prompt": "a potted fern", PREFER: op.model}
    classical_op = op.model_copy(update={"model": "classical"})
    assert PREFER not in job_params(classical_op)


def test_manual_generative_edits_are_stamped(client: TestClient, upload: Upload) -> None:
    doc = upload("landscape.png")
    state = EditState(layers=[fill(seed=None)]).model_dump(mode="json")
    res = client.post(
        f"/api/documents/{doc['id']}/edits", json={"label": "Add a plant", "state": state}
    )
    assert res.status_code == 200, res.text
    (op,) = res.json()["state"]["layers"][0]["operations"]
    assert op["model"] == "classical" and isinstance(op["seed"], int)
    bad = EditState(layers=[fill()]).model_dump(mode="json")
    bad["layers"][0]["operations"].append({"op": "exposure", "stops": 1})
    res = client.post(f"/api/documents/{doc['id']}/edits", json={"label": "x", "state": bad})
    assert res.status_code == 422


def test_the_agent_generates_in_its_own_masked_layer() -> None:
    editor = Editor(EditState(), model_for=lambda task: "classical")
    with pytest.raises(ToolError, match="own new layer"):
        editor.call("generate", {"prompt": "a fern"})
    editor.call("add_layer", {"name": "Add a fern", "mask": SPOT.model_dump()})
    text, event = editor.call("generate", {"prompt": "a fern in a pot"})
    assert "no generative model is installed" in text
    assert event.summary == "Generate “a fern in a pot”"
    with pytest.raises(ToolError, match="changes what is in the photo"):
        editor.call("contrast", {"amount": 10})
    (op,) = editor.state.layers[-1].operations
    assert isinstance(op, ops.Generate) and op.seed is not None and op.model == "classical"
    # The agent never sees or sets the recorded model.
    tool = next(t for t in TOOLS if t["name"] == "generate")
    assert set(tool["input_schema"]["properties"]) == {"prompt", "seed", "grow"}  # type: ignore[call-overload]


def test_recipes_leave_out_generative_layers_tied_to_one_photo() -> None:
    sky = {"kind": "semantic", "target": "sky"}
    dog = {"kind": "semantic", "target": "object", "box": [0.1, 0.1, 0.4, 0.5]}
    kept = portable([fill(mask=sky), fill(mask=dog)])
    assert len(kept) == 1 and kept[0].mask is not None


class Recorder:
    uses_weights = True
    license = "test"

    def __init__(self, name: str) -> None:
        self.name = name

    def available(self) -> bool:
        return True

    def run(self, image: Any, params: dict[str, Any], progress: Progress, env: Env) -> Any:
        assert PREFER not in params
        return self.name


def test_the_worker_tries_the_recorded_model_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(TASKS, "two", Task("two", "Testing", [Recorder("a"), Recorder("b")]))
    worker = ModelWorker(WorkerConfig(device="cpu", cache_dir=tmp_path), "inline")
    try:
        assert worker.run("two").value == "a"
        assert worker.run("two", params={PREFER: "b"}).value == "b"
        assert worker.run("two", params={PREFER: "gone"}).value == "a"
    finally:
        worker.close()


# Expanding the canvas


def test_expand_box_grows_to_an_aspect_or_by_sides() -> None:
    assert expand_box(ops.Expand(aspect="16:9"), 300, 400) == (711, 400, 205, 0)
    assert expand_box(ops.Expand(aspect="1:1"), 400, 300) == (400, 400, 0, 50)
    assert expand_box(ops.Expand(aspect="4:5"), 400, 600) == (480, 600, 40, 0)
    assert expand_box(ops.Expand(left=0.5, bottom=0.25), 200, 100) == (300, 125, 100, 0)


def test_extend_canvas_keeps_the_photo_and_mirrors_softly() -> None:
    rng = np.random.default_rng(0)
    photo = rng.random((40, 60, 3)).astype(np.float32)
    out = extend_canvas(photo, 100, 40, 20, 0)
    assert out.shape == (40, 100, 3)
    np.testing.assert_array_equal(out[:, 20:80], photo)
    # Next to the photo it mirrors the photo; far away it is softened.
    np.testing.assert_allclose(out[:, 19], photo[:, 0], atol=0.05)
    assert out[:, :3].std() < photo.std()


def test_expanding_the_canvas_paints_new_surroundings(settings: Settings, upload: Upload) -> None:
    doc = upload("portrait.jpg")
    loaded = get_store(settings).image(doc["id"])
    plain = render_state(loaded.proxy, EditState(), loaded.proxy_context)
    wide = EditState.model_validate({"framing": [ops.Expand(aspect="16:9", seed=1)]})
    out = render_state(loaded.proxy, wide, loaded.proxy_context)
    h, w = plain.shape[:2]
    assert out.shape[0] == h and abs(out.shape[1] / h - 16 / 9) < 0.01
    x0 = (out.shape[1] - w) // 2
    middle = out[:, x0 + w // 4 : x0 + 3 * w // 4]
    np.testing.assert_allclose(middle, plain[:, w // 4 : 3 * w // 4], atol=1e-5)
    assert loaded.vision is not None
    tasks = [j.task for j in loaded.vision.worker.jobs(doc["id"])]
    assert tasks == ["generate"]
    # The export is the same shape, from the same generated result.
    full = render_state(loaded.source.pixels, wide, loaded.full_context)
    assert abs(full.shape[1] / full.shape[0] - 16 / 9) < 0.01
    assert len(loaded.vision.worker.jobs(doc["id"])) == 1


def test_the_agent_expands_the_framing() -> None:
    editor = Editor(EditState(), model_for=lambda task: "sdxl-inpainting-0.1")
    text, _ = editor.call("expand", {"aspect": "16:9"})
    assert "the framing" in text and "no generative model" not in text
    (op,) = editor.state.framing
    assert isinstance(op, ops.Expand) and op.seed is not None
    assert op.model == "sdxl-inpainting-0.1"


# Replacing the background


def test_harmonize_moves_the_subject_toward_the_scene() -> None:
    image = np.zeros((60, 60, 3), np.float32)
    image[:] = [0.2, 0.3, 0.8]  # a blue scene
    image[20:40, 20:40] = [0.8, 0.6, 0.4]  # a warm subject
    subject = np.zeros((60, 60), np.float32)
    subject[20:40, 20:40] = 1
    out = harmonize(image, subject, 100)
    np.testing.assert_allclose(out[subject == 0], image[subject == 0], atol=1e-6)
    before, after = image[30, 30], out[30, 30]
    assert after[2] - after[0] > before[2] - before[0]  # bluer
    assert out[20, 30, 2] > after[2] + 0.03  # the scene's light wraps over the edge
    assert harmonize(image, subject, 0) is image


def replacement(**op: Any) -> Layer:
    return Layer(
        id="Lbg",
        name="Beach",
        operations=[ops.ReplaceBackground(prompt="a sunny beach", seed=3, **op)],
    )


def test_replacing_the_background_keeps_the_subject(settings: Settings, upload: Upload) -> None:
    doc = upload("portrait.jpg")
    loaded = get_store(settings).image(doc["id"])
    ctx = loaded.proxy_context
    before = render_state(loaded.proxy, EditState(), ctx)
    plain = render_state(loaded.proxy, EditState(layers=[replacement(harmonize=0)]), ctx)
    subject = render_layer_mask(
        loaded.proxy,
        EditState(layers=[replacement()]),
        "Lbg",
        ctx,
    )
    assert subject.mean() > 0.1  # the default mask: everything but the subject
    keep = subject < 0.02
    background = subject > 0.98
    assert np.abs(plain - before)[keep].max() < 0.02
    assert np.abs(plain - before)[background].mean() > 0.02
    assert loaded.vision is not None
    tasks = [j.task for j in loaded.vision.worker.jobs(doc["id"])]
    assert tasks == ["segment_subject", "generate"]
    matched = render_state(loaded.proxy, EditState(layers=[replacement(harmonize=100)]), ctx)
    assert np.abs(matched - plain)[keep].mean() > 0.002
    assert len(loaded.vision.worker.jobs(doc["id"])) == 2  # harmonizing needs no model


def test_the_agent_replaces_the_background_in_a_layer_of_its_own() -> None:
    editor = Editor(EditState())
    editor.call("add_layer", {"name": "Warmer"})
    editor.call("white_balance", {"temperature": 20})
    editor.call("replace_background", {"prompt": "a misty forest"})
    warm, scene = editor.state.layers
    assert warm.name == "Warmer" and len(warm.operations) == 1
    assert scene.is_content and scene.mask is None
    assert scene.name == "Replace background “a misty forest”"


def test_classical_relight_lights_the_chosen_side() -> None:
    image = np.full((60, 80, 3), 0.5, np.float32)
    lit = classical.relight(image, direction="left", prompt="")
    assert lit[:, :20].mean() > lit[:, -20:].mean() + 0.05
    warm = classical.relight(image, direction="front", prompt="warm sunset light")
    assert warm[..., 0].mean() > warm[..., 2].mean() + 0.05
    assert classical.light_color("cool moonlight")[2] == 1.0


def test_transfer_light_keeps_the_photos_detail() -> None:
    rng = np.random.default_rng(0)
    photo = np.clip(0.5 + 0.1 * rng.standard_normal((80, 80, 3)), 0, 1).astype(np.float32)
    lit = np.full_like(photo, 0.25)  # same scene, half the light, no detail
    out = transfer_light(photo, lit)
    assert abs(out.mean() - 0.26) < 0.03
    assert np.corrcoef(out[..., 0].ravel(), photo[..., 0].ravel())[0, 1] > 0.9


def relighting(**op: Any) -> Layer:
    return Layer(
        id="Llight",
        name="Sunset",
        operations=[ops.Relight(direction="right", prompt="warm sunset", seed=5, **op)],
    )


def test_relighting_changes_the_light(settings: Settings, upload: Upload) -> None:
    doc = upload("portrait.jpg")
    loaded = get_store(settings).image(doc["id"])
    ctx = loaded.proxy_context
    before = render_state(loaded.proxy, EditState(), ctx)
    after = render_state(loaded.proxy, EditState(layers=[relighting(amount=100)]), ctx)
    w = before.shape[1]
    gain = after.mean(axis=2) - before.mean(axis=2)
    assert gain[:, -w // 4 :].mean() > gain[:, : w // 4].mean() + 0.03
    assert (after[..., 0] - after[..., 2]).mean() > (before[..., 0] - before[..., 2]).mean()
    half = render_state(loaded.proxy, EditState(layers=[relighting(amount=50)]), ctx)
    assert np.abs(half - (before + after) / 2).mean() < 0.01
    assert loaded.vision is not None
    assert [j.task for j in loaded.vision.worker.jobs(doc["id"])] == ["relight"]


def test_the_agent_relights_in_a_layer_of_its_own() -> None:
    editor = Editor(EditState())
    editor.call("relight", {"direction": "top", "prompt": "soft window light"})
    (layer,) = editor.state.layers
    assert layer.is_content and layer.mask is None
    op = layer.operations[0]
    assert isinstance(op, ops.Relight) and op.seed is not None
