"""Auto-straighten and lens correction: finding tilt and converging verticals, and the
perspective and distortion fixes."""

from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from photo_agent import imaging
from photo_agent import operations as ops
from photo_agent.agent import Editor, ToolError
from photo_agent.geometry import AutoStraighten, auto_level, keystone, tilt
from photo_agent.layers import EditState
from photo_agent.render import RenderContext, perspective_quad, render

CTX = RenderContext()


def scene(w: int = 600, h: int = 400) -> imaging.Array:
    """A row of pale columns with dark rails across them, all level and plumb."""
    img = np.full((h, w, 3), 0.6, np.float32)
    img[int(h * 0.55) :] = 0.3
    for x in range(60, w, 70):
        cv2.rectangle(img, (x, int(h * 0.15)), (x + 25, int(h * 0.9)), (0.9, 0.85, 0.8), -1)
    for y in range(int(h * 0.2), int(h * 0.85), 45):
        cv2.line(img, (30, y), (w - 30, y), (0.1, 0.1, 0.15), 2)
    return img


def converging(img: imaging.Array, vertical: float) -> imaging.Array:
    """What a camera tilted up or down sees: the inverse of Perspective(vertical)."""
    h, w = img.shape[:2]
    frame = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    quad = perspective_quad(ops.Perspective(vertical=vertical), w, h)
    matrix = cv2.getPerspectiveTransform(frame, quad)
    out = cv2.warpPerspective(img, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return out.astype(np.float32)


@pytest.mark.parametrize("angle", [-6.0, -2.5, 3.0, 8.0])
def test_tilt_finds_the_straighten_that_undoes_it(angle: float) -> None:
    tilted = render(scene(), [ops.Straighten(angle=angle)], CTX)
    found = tilt(tilted)
    assert found is not None
    assert found == pytest.approx(-angle, abs=0.1)


@pytest.mark.parametrize("vertical", [-40.0, 20.0, 50.0])
def test_keystone_finds_the_perspective_that_undoes_it(vertical: float) -> None:
    found = keystone(converging(scene(), vertical))
    assert found == pytest.approx(vertical, abs=2)
    fixed = render(converging(scene(), vertical), [ops.Perspective(vertical=vertical)], CTX)
    assert abs(keystone(fixed) or 0) < 2


def test_nothing_to_go_by() -> None:
    flat = np.full((300, 400, 3), 0.5, np.float32)
    assert tilt(flat) is None
    assert keystone(flat) is None
    assert keystone(scene()) == pytest.approx(0, abs=0.5)
    found = auto_level([], lambda f: render(flat, f, CTX))
    assert found.framing == [] and found.describe() == ""


@pytest.mark.parametrize("name", ["phone.heic", "low-light.jpg"])
def test_photos_without_lines_are_left_alone(photos: Path, name: str) -> None:
    px = imaging.make_proxy(imaging.decode((photos / name).read_bytes()).pixels)
    assert auto_level([], lambda f: render(px, f, CTX)).describe() == ""


def test_auto_level_replaces_earlier_fixes_and_keeps_the_crop() -> None:
    img = converging(scene(), 30)
    crop = ops.Crop(left=0.1, right=0.9, aspect="1:1")
    flip = ops.Flip()
    framing: list[ops.OpBase] = [
        ops.Straighten(angle=-3),
        flip,
        ops.Perspective(vertical=-10),
        crop,
    ]
    found = auto_level(framing, lambda f: render(img, f, CTX))
    assert [op.op for op in found.framing] == ["flip", "perspective", "crop"]  # type: ignore[attr-defined]
    assert found.framing[0] is flip and found.framing[2] is crop
    assert found.framing[1].vertical == pytest.approx(30, abs=2)  # type: ignore[attr-defined]
    assert found.describe().startswith("perspective vertical +")
    only_level = auto_level(
        framing, lambda f: render(img, f, CTX), AutoStraighten(perspective=False)
    )
    assert [op.op for op in only_level.framing] == ["flip", "perspective", "crop"]  # type: ignore[attr-defined]
    assert only_level.framing[1].vertical == -10  # type: ignore[attr-defined]


def first_column(img: imaging.Array) -> int:
    return int(np.argmax(img[int(img.shape[0] * 0.2), :, 0] > 0.8))


def test_perspective_widens_the_narrow_side_without_empty_corners() -> None:
    img = np.full((300, 400, 3), 0.7, np.float32)
    out = render(img, [ops.Perspective(vertical=60, horizontal=-30)], CTX)
    assert out.shape == img.shape
    np.testing.assert_allclose(out, 0.7, atol=1e-3)  # every pixel comes from the photo
    lines = render(scene(), [ops.Perspective(vertical=60)], CTX)
    # The top is stretched out: the first column starts further left than it did.
    assert first_column(lines) < first_column(scene())


def test_lens_correction_bends_lines_and_fills_the_frame() -> None:
    h, w = 400, 600
    img = np.full((h, w, 3), 0.8, np.float32)
    cv2.line(img, (0, 60), (w, 60), (0.0, 0.0, 0.0), 3)

    def row_of_line(out: imaging.Array, x: int) -> float:
        return float(np.argmin(out[:, x, 0]))

    barrel_fix = render(img, [ops.LensCorrection(distortion=60)], CTX)
    # Correcting barrel pulls the corners out, so a straight line near the top sags.
    assert row_of_line(barrel_fix, 20) < row_of_line(barrel_fix, w // 2) - 2
    pin_fix = render(img, [ops.LensCorrection(distortion=-60)], CTX)
    assert row_of_line(pin_fix, 20) > row_of_line(pin_fix, w // 2) + 2
    for out in (barrel_fix, pin_fix):
        assert out.shape == img.shape
        np.testing.assert_allclose(out[h // 2, [0, -1]], 0.8, atol=0.02)


def test_the_agent_straightens() -> None:
    img = render(scene(), [ops.Straighten(angle=4)], CTX)
    editor = Editor(EditState(), framed=lambda f: render(img, f, CTX))
    text, _ = editor.call("auto_straighten", {})
    assert "straighten -4" in text
    assert [op.op for op in editor.state.framing] == ["straighten"]
    flat = np.full((300, 400, 3), 0.5, np.float32)
    with pytest.raises(ToolError, match="nothing changed"):
        Editor(EditState(), framed=lambda f: render(flat, f, CTX)).call("auto_straighten", {})
    with pytest.raises(ToolError, match="not available"):
        Editor(EditState()).call("auto_straighten", {})


def test_auto_straighten_endpoint(client: TestClient) -> None:
    img = render(scene(), [ops.Straighten(angle=-5)], CTX)
    res = client.post("/api/documents", files={"file": ("tilted.png", imaging.encode_png(img))})
    doc = res.json()
    res = client.post(f"/api/documents/{doc['id']}/straighten")
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["undo_label"] == "Auto straighten"
    (op,) = body["state"]["framing"]
    assert op["op"] == "straighten" and op["angle"] == pytest.approx(5, abs=0.2)

    flat = imaging.encode_png(np.full((300, 400, 3), 0.5, np.float32))
    doc = client.post("/api/documents", files={"file": ("flat.png", flat)}).json()
    res = client.post(f"/api/documents/{doc['id']}/straighten")
    assert res.status_code == 422
