"""Background removal: cutouts with transparent or colored backgrounds."""

import io
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import ValidationError

from photo_agent.agent import Editor, ToolError, describe_state
from photo_agent.layers import Cutout, EditState
from photo_agent.masks import RadialGradientMask, SemanticMask
from photo_agent.render import CHECKER, render_cutout, render_state
from photo_agent.routes import get_store
from photo_agent.settings import Settings

Upload = Callable[[str], dict[str, Any]]
MIDDLE = RadialGradientMask(center=[0.5, 0.5], radius_x=0.3, radius_y=0.3, feather=0)


def test_cutout_defaults_to_the_subject_on_transparency() -> None:
    cutout = Cutout()
    assert cutout.mask == SemanticMask(target="subject")
    assert cutout.background is None
    with pytest.raises(ValidationError):
        Cutout(background="white")


def test_cutout_alpha_and_preview_backdrop(settings: Settings, upload: Upload) -> None:
    doc = upload("portrait.jpg")
    loaded = get_store(settings).image(doc["id"])
    ctx = loaded.proxy_context
    plain = render_state(loaded.proxy, EditState(), ctx)

    state = EditState(cutout=Cutout())
    rgb, alpha = render_cutout(loaded.proxy, state, ctx)
    assert alpha is not None
    np.testing.assert_allclose(rgb, plain)
    assert alpha[80:150, 190:260].mean() > 0.8  # the astronaut's face
    assert alpha[:40, 440:].mean() < 0.2  # the backdrop

    preview = render_state(loaded.proxy, state, ctx)
    # Where the backdrop is cut away, the preview shows the gray checkerboard.
    corner = preview[:20, 490:]
    np.testing.assert_allclose(corner[..., 0], corner[..., 2], atol=0.02)
    assert min(CHECKER) - 0.02 <= corner.min() and corner.max() <= max(CHECKER) + 0.02

    white = render_state(loaded.proxy, EditState(cutout=Cutout(background="#ffffff")), ctx)
    assert white[:40, 440:].min() > 0.8
    hidden = EditState(cutout=Cutout(visible=False))
    np.testing.assert_allclose(render_state(loaded.proxy, hidden, ctx), plain)
    assert render_cutout(loaded.proxy, hidden, ctx)[1] is None


def set_state(settings: Settings, doc_id: str, state: EditState) -> None:
    store = get_store(settings)
    doc = store.get(doc_id)
    doc.edit_by_hand("test", state)
    store.save(doc)


def test_export_transparent_png_or_white_jpeg(
    client: TestClient, upload: Upload, settings: Settings
) -> None:
    doc = upload("portrait.jpg")
    set_state(settings, doc["id"], EditState(cutout=Cutout(mask=MIDDLE)))
    res = client.post(f"/api/documents/{doc['id']}/export", json={"format": "png"})
    img = Image.open(io.BytesIO(res.content))
    assert img.mode == "RGBA"
    alpha = np.asarray(img.getchannel("A"))
    assert alpha[256, 256] == 255 and alpha[5, 5] == 0

    res = client.post(f"/api/documents/{doc['id']}/export", json={"format": "jpeg"})
    jpeg = np.asarray(Image.open(io.BytesIO(res.content)).convert("RGB"))
    assert jpeg[5, 5].min() > 245


def test_the_cutout_mask_can_be_shown(
    client: TestClient, upload: Upload, settings: Settings
) -> None:
    doc = upload("portrait.jpg")
    url = f"/api/documents/{doc['id']}/layers/cutout/mask"
    assert client.get(url).status_code == 404
    set_state(settings, doc["id"], EditState(cutout=Cutout(mask=MIDDLE, visible=False)))
    alpha = np.asarray(Image.open(io.BytesIO(client.get(url).content)))
    assert alpha[256, 256] == 255 and alpha[5, 5] == 0


def test_the_agent_cuts_out_and_restores() -> None:
    editor = Editor(EditState())
    editor.call("cut_out", {"background": "#ffffff"})
    assert editor.state.cutout == Cutout(background="#ffffff")
    assert "keeps the semantic mask selecting subject, #ffffff background" in describe_state(
        editor.state
    )
    with pytest.raises(ToolError, match="Invalid cutout"):
        editor.call("cut_out", {"background": "blue"})
    editor.call("restore_background", {})
    assert editor.state.cutout is None
    assert editor.state.is_empty
    with pytest.raises(ToolError):
        editor.call("restore_background", {})
