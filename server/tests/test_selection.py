"""Semantic masks: the classical fallbacks, caching, and selections through the API."""

import io
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import ValidationError

from photo_agent import imaging
from photo_agent import operations as ops
from photo_agent.agent import Editor
from photo_agent.layers import EditState, Layer
from photo_agent.masks import MaskUnavailableError, SemanticMask, describe_mask, render_mask
from photo_agent.recipes import portable
from photo_agent.render import RenderContext, render_state
from photo_agent.routes import get_store
from photo_agent.settings import Settings
from photo_agent.vision import classical

Upload = Callable[[str], dict[str, Any]]


def photo(photos: Path, name: str) -> imaging.Array:
    return imaging.make_proxy(imaging.decode((photos / name).read_bytes()).pixels)


def test_classical_sky_finds_the_sky(photos: Path) -> None:
    land = photo(photos, "landscape.png")
    sky = classical.sky(land)
    assert sky.shape == land.shape[:2]
    h, w = sky.shape
    assert sky[: h // 4, w // 4 : w // 2].mean() > 0.9
    assert sky[-h // 10 :].mean() < 0.3


def test_classical_subject_keeps_the_middle(photos: Path) -> None:
    portrait = photo(photos, "portrait.jpg")
    subject = classical.subject(portrait)
    assert subject[80:150, 190:260].mean() > 0.8  # face
    assert subject[300:400, 100:200].mean() > 0.8  # suit
    assert subject[:40, 420:].mean() < 0.2


def test_classical_object_follows_points(photos: Path) -> None:
    cat = photo(photos, "phone.heic")
    found = classical.object_at(cat, None, [[0.6, 0.5, True]])
    h, w = found.shape
    assert found[h // 2, round(w * 0.6)] > 0.9
    assert classical.object_at(cat, None, []).max() == 0.0


def test_classical_face_parts_find_a_face(photos: Path) -> None:
    portrait = photo(photos, "portrait.jpg")
    (face,) = classical.find_faces(portrait)
    x0, y0, x1, y1 = face
    assert 150 < x0 < x1 < 300 and 30 < y0 < y1 < 220
    parts = classical.face_parts(portrait)
    assert set(parts) == set(classical.FACE_PARTS)
    inside = parts["skin"][round(y0) : round(y1), round(x0) : round(x1)]
    assert inside.mean() > 0.4
    assert parts["skin"][400:, :].mean() < 0.05
    assert parts["eyes"].sum() < parts["skin"].sum()


def test_object_selection_needs_a_box_or_points() -> None:
    with pytest.raises(ValidationError, match="box or points"):
        SemanticMask(target="object")
    with pytest.raises(ValidationError, match="right > left"):
        SemanticMask(target="object", box=[0.5, 0.1, 0.4, 0.9])
    mask = SemanticMask(target="object", box=[0.1, 0.1, 0.4, 0.9], description="the dog")
    assert describe_mask(mask) == "semantic mask selecting the dog in box (0.10, 0.10)-(0.40, 0.90)"
    assert describe_mask(SemanticMask(target="sky", invert=True)) == (
        "inverted semantic mask selecting sky"
    )


def test_semantic_masks_need_the_model_worker() -> None:
    with pytest.raises(MaskUnavailableError):
        render_mask(SemanticMask(target="sky"), np.zeros((4, 4, 3), np.float32))


def sky_state(**mask: Any) -> EditState:
    layer = Layer(
        id="Lsky",
        name="Darker sky",
        mask=SemanticMask(target="sky", **mask),
        operations=[ops.Exposure(stops=-1.5)],
    )
    return EditState(layers=[layer])


def test_a_sky_layer_changes_only_the_sky(settings: Settings, upload: Upload) -> None:
    doc = upload("landscape.png")
    loaded = get_store(settings).image(doc["id"])
    before = render_state(loaded.proxy, EditState(), loaded.proxy_context)
    after = render_state(loaded.proxy, sky_state(), loaded.proxy_context)
    h = before.shape[0]
    assert after[: h // 4].mean() < before[: h // 4].mean() * 0.6
    np.testing.assert_allclose(after[-h // 12 :, :40], before[-h // 12 :, :40], atol=0.05)
    inverted = render_state(loaded.proxy, sky_state(invert=True), loaded.proxy_context)
    np.testing.assert_allclose(inverted[: h // 5, 100:300], before[: h // 5, 100:300], atol=0.05)


def test_selections_are_cached_by_what_they_depend_on(settings: Settings, upload: Upload) -> None:
    doc = upload("landscape.png")
    store = get_store(settings)
    loaded = store.image(doc["id"])
    vision = loaded.vision
    assert vision is not None
    sky = SemanticMask(target="sky")
    first = vision.selection(sky, [])
    jobs = len(vision.worker.jobs(doc["id"]))
    np.testing.assert_array_equal(vision.selection(sky, []), first)
    np.testing.assert_array_equal(
        vision.selection(sky.model_copy(update={"invert": True}), []), first
    )
    assert len(vision.worker.jobs(doc["id"])) == jobs
    assert list((store.root / doc["id"] / "vision").glob("*.png"))

    crop = [ops.Crop(left=0.2, right=0.8)]
    assert vision.key(sky, crop) != vision.key(sky, [])
    assert vision.key(sky, crop) == vision.key(sky, [ops.Crop(left=0.2, right=0.8)])
    assert vision.selection(sky, crop).shape[1] < first.shape[1]
    one = SemanticMask(target="object", box=[0.1, 0.1, 0.3, 0.9])
    other = SemanticMask(target="object", box=[0.5, 0.1, 0.7, 0.9])
    assert vision.key(one, []) != vision.key(other, [])
    # Every face part comes from one face-parsing job.
    eyes, lips = SemanticMask(target="eyes"), SemanticMask(target="lips")
    assert vision.key(eyes, []) == vision.key(lips, [])


def test_editing_a_semantic_mask_through_the_api(client: TestClient, upload: Upload) -> None:
    doc = upload("landscape.png")
    state = sky_state().model_dump(mode="json")
    res = client.post(
        f"/api/documents/{doc['id']}/edits", json={"label": "Darker sky", "state": state}
    )
    assert res.status_code == 200, res.text
    jobs = client.get(f"/api/documents/{doc['id']}/jobs").json()
    assert [(j["task"], j["state"], j["backend"]) for j in jobs] == [
        ("segment_sky", "done", "classical")
    ]
    mask = client.get(f"/api/documents/{doc['id']}/layers/Lsky/mask").content
    alpha = np.asarray(Image.open(io.BytesIO(mask)), np.float32) / 255
    assert alpha[: alpha.shape[0] // 4].mean() > 0.75
    assert client.get(f"/api/documents/{doc['id']}/preview").status_code == 200


def test_the_agent_can_select_things() -> None:
    editor = Editor(EditState())
    editor.call(
        "add_layer",
        {
            "name": "Brighter dog",
            "mask": {
                "kind": "semantic",
                "target": "object",
                "box": [0.2, 0.3, 0.6, 0.9],
                "points": [{"x": 0.4, "y": 0.6}],
                "description": "the dog",
            },
        },
    )
    (layer,) = editor.state.layers
    assert isinstance(layer.mask, SemanticMask)
    assert layer.mask.points[0].include


def test_recipes_keep_portable_selections() -> None:
    layers = [
        Layer(id="L1", name="Sky", mask=SemanticMask(target="sky")),
        Layer(id="L2", name="Dog", mask=SemanticMask(target="object", box=[0, 0, 0.5, 0.5])),
    ]
    sky, dog = portable(layers)
    assert sky.mask == SemanticMask(target="sky")
    assert dog.mask is None


def test_render_context_without_vision_still_renders_other_masks() -> None:
    out = render_state(np.full((8, 8, 3), 0.5, np.float32), EditState(), RenderContext())
    assert out.shape == (8, 8, 3)
