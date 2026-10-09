"""Looks (ready-made color styles) and restyling."""

from collections.abc import Callable
from typing import Any

import numpy as np
from fastapi.testclient import TestClient

from photo_agent import operations as ops
from photo_agent.agent import TOOLS, Editor, ToolError
from photo_agent.layers import EditState, Layer
from photo_agent.looks import LOOKS
from photo_agent.render import RenderContext, hue_tint, render_state
from photo_agent.routes import get_store
from photo_agent.settings import Settings
from photo_agent.vision import classical

Upload = Callable[[str], dict[str, Any]]


def test_hue_tints_keep_brightness() -> None:
    for hue in (0, 30, 120, 210, 300):
        tint = hue_tint(hue)
        assert abs(float(tint @ classical.LUMA)) < 1e-5
    assert hue_tint(30)[0] > 0 > hue_tint(30)[2]  # orange: more red, less blue


def test_color_grade_tints_shadows_and_highlights_apart() -> None:
    ramp = np.repeat(np.linspace(0, 1, 64, dtype=np.float32)[None, :, None], 8, 0)
    image = np.repeat(ramp, 3, 2)
    grade = ops.ColorGrade(shadows_hue=210, shadows=80, highlights_hue=30, highlights=80)
    out = render_state(
        image, EditState(layers=[Layer(id="L", name="g", operations=[grade])]), RenderContext()
    )
    dark, light = out[:, 12], out[:, 52]
    assert (dark[:, 2] > dark[:, 0]).all()  # cool shadows
    assert (light[:, 0] > light[:, 2]).all()  # warm highlights


def test_every_look_renders(settings: Settings, upload: Upload) -> None:
    doc = upload("landscape.png")
    loaded = get_store(settings).image(doc["id"])
    before = render_state(loaded.proxy, EditState(), loaded.proxy_context)
    assert len({look.id for look in LOOKS}) == len(LOOKS) >= 8
    for look in LOOKS:
        state = EditState(layers=[Layer(id="L", name=look.name, operations=look.operations)])
        after = render_state(loaded.proxy, state, loaded.proxy_context)
        assert np.isfinite(after).all()
        assert np.abs(after - before).mean() > 0.005, look.id


def test_looks_api(client: TestClient, upload: Upload) -> None:
    listed = client.get("/api/looks").json()
    assert [look["id"] for look in listed] == [look.id for look in LOOKS]
    doc = upload("landscape.png")
    res = client.post(f"/api/documents/{doc['id']}/looks/noir", json={"strength": 60})
    assert res.status_code == 200
    (layer,) = res.json()["state"]["layers"]
    assert layer["name"] == "Noir" and layer["opacity"] == 60
    assert res.json()["history"][-1]["label"] == "Look: Noir"
    assert client.post(f"/api/documents/{doc['id']}/looks/nope").status_code == 404


def test_the_agent_applies_a_look() -> None:
    tool = next(t for t in TOOLS if t["name"] == "apply_look")
    schema: Any = tool["input_schema"]
    assert "teal-orange" in schema["properties"]["look"]["enum"]
    editor = Editor(EditState())
    editor.call("apply_look", {"look": "film-70s", "strength": 80})
    (layer,) = editor.state.layers
    assert layer.name == "1970s film" and layer.opacity == 80
    assert any(isinstance(op, ops.ColorGrade) for op in layer.operations)
    try:
        editor.call("apply_look", {"look": "sepia-dreams"})
    except ToolError as exc:
        assert "Unknown look" in str(exc)
    else:
        raise AssertionError("expected an unknown look to fail")


def test_restyle_redraws_the_photo(settings: Settings, upload: Upload) -> None:
    doc = upload("landscape.png")
    loaded = get_store(settings).image(doc["id"])
    before = render_state(loaded.proxy, EditState(), loaded.proxy_context)
    layer = Layer(id="Ls", name="Watercolor", operations=[ops.Restyle(prompt="watercolor", seed=2)])
    after = render_state(loaded.proxy, EditState(layers=[layer]), loaded.proxy_context)
    assert np.abs(after - before).mean() > 0.01
    assert loaded.vision is not None
    assert [j.task for j in loaded.vision.worker.jobs(doc["id"])] == ["generate"]


def test_classical_backdrop_and_stylize() -> None:
    image = np.random.default_rng(0).random((40, 60, 3)).astype(np.float32)
    mask = np.zeros((40, 60), np.float32)
    mask[:, :30] = 1
    plain = classical.generate(image, mask=mask, backdrop=True)
    assert plain[:, :30].std() < 0.05  # a smooth backdrop, not the old noise
    np.testing.assert_array_equal(plain[:, 30:], image[:, 30:])
    painted = classical.generate(image, restyle=True, strength=0.8)
    assert painted.shape == image.shape and painted.std() < image.std()
