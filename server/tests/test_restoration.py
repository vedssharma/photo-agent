"""Restoration: enlarging on export, restoring faces, and colorizing black-and-white photos."""

import io
from collections.abc import Callable
from typing import Any

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from photo_agent import operations as ops
from photo_agent.agent import Editor
from photo_agent.layers import EditState, Layer
from photo_agent.render import keep_luminance, render_state
from photo_agent.routes import get_store
from photo_agent.settings import Settings
from photo_agent.vision import classical

Upload = Callable[[str], dict[str, Any]]


def test_classical_upscale_enlarges_8_bit_pixels() -> None:
    image = (np.random.default_rng(0).random((20, 30, 3)) * 255).astype(np.uint8)
    big = classical.upscale(image, scale=2.5)
    assert big.dtype == np.uint8 and big.shape == (50, 75, 3)
    assert abs(float(big.mean()) - float(image.mean())) < 3


def test_export_can_be_enlarged(client: TestClient, upload: Upload) -> None:
    doc = upload("landscape.png")
    plain = Image.open(
        io.BytesIO(client.post(f"/api/documents/{doc['id']}/export", json={}).content)
    )
    res = client.post(f"/api/documents/{doc['id']}/export", json={"upscale": 2, "format": "png"})
    assert res.status_code == 200
    big = Image.open(io.BytesIO(res.content))
    assert big.size == (plain.size[0] * 2, plain.size[1] * 2)
    bad = client.post(f"/api/documents/{doc['id']}/export", json={"upscale": 3})
    assert bad.status_code == 422


def test_keep_luminance_takes_only_color() -> None:
    photo = np.repeat(np.linspace(0.1, 0.9, 40, dtype=np.float32)[None, :, None], 30, 0)
    photo = np.repeat(photo, 3, 2)
    colored = np.zeros_like(photo)
    colored[...] = [0.6, 0.45, 0.35]
    out = keep_luminance(photo, colored)
    assert (out[..., 0] > out[..., 2] + 0.05).mean() > 0.8
    assert np.abs(classical.gray(out) - classical.gray(photo)).mean() < 0.05


def test_colorizing_adds_color_and_keeps_the_photo(settings: Settings, upload: Upload) -> None:
    doc = upload("landscape.png")
    loaded = get_store(settings).image(doc["id"])
    ctx = loaded.proxy_context
    mono = EditState(layers=[Layer(id="Lbw", name="B&W", operations=[ops.Saturation(amount=-100)])])
    gray = render_state(loaded.proxy, mono, ctx)
    layer = Layer(id="Lc", name="Colorize", operations=[ops.Colorize(seed=1)])
    restored = render_state(gray, EditState(layers=[layer]), ctx)
    chroma = restored.max(axis=2) - restored.min(axis=2)
    assert chroma.mean() > 0.03
    assert np.abs(classical.gray(restored) - classical.gray(gray)).mean() < 0.04


def test_classical_face_restoration_changes_only_faces() -> None:
    image = np.full((120, 160, 3), [0.3, 0.5, 0.3], np.float32)
    out = classical.restore_faces(image)
    np.testing.assert_array_equal(out, image)  # no faces, nothing to do


def test_the_agent_restores_and_colorizes_in_layers_of_their_own() -> None:
    editor = Editor(EditState())
    editor.call("restore_faces", {})
    editor.call("colorize", {"amount": 80})
    faces, color = editor.state.layers
    assert faces.name == "Restore faces" and faces.is_content
    assert color.name == "Colorize" and isinstance(color.operations[0], ops.Colorize)
