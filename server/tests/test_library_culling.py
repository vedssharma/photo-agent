"""Picking the best shots: quality measures, moments, picks, and their explanations."""

import io
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fake_advisor import FakeAdvisor
from fastapi.testclient import TestClient
from library_helpers import face, import_bytes, jpeg_with_exif
from PIL import Image, ImageFilter

from photo_agent.advisor import AdvisorError
from photo_agent.imaging import Array
from photo_agent.main import app
from photo_agent.routes import get_advisor
from photo_agent.vision import quality


def pixels(data: bytes) -> Array:
    rgb = Image.open(io.BytesIO(data)).convert("RGB")
    return (np.asarray(rgb, np.float32) / 255).astype(np.float32)


def test_quality_measures(photos: Path) -> None:
    landscape = Image.open(photos / "landscape.png").convert("RGB")
    sharp = quality.measure(pixels((photos / "landscape.png").read_bytes()))
    blurred = landscape.filter(ImageFilter.GaussianBlur(3))
    soft = quality.measure(pixels(jpeg_with_exif(image=blurred)))
    assert sharp.sharpness > 0.8 > 0.4 > soft.sharpness

    blown = np.clip(pixels((photos / "landscape.png").read_bytes()) * 2.5, 0, 1)
    assert quality.measure(blown).exposure < sharp.exposure - 0.3

    awake = quality.measure(pixels(face("a", (0.2, 0.3, 0.6))))
    blink = quality.measure(pixels(face("a", (0.2, 0.3, 0.6), eyes="closed")))
    assert (awake.faces, awake.eyes_closed) == (1, 0)
    assert (blink.faces, blink.eyes_closed) == (1, 1)


@pytest.fixture
def shoot(client: TestClient, photos: Path) -> dict[str, str]:
    """A day out in three moments: a burst of a view (one frame sharp, two shaken), two
    frames of a portrait (one blinking), and a single night shot."""
    view = Image.open(photos / "landscape.png")
    sitter = Image.open(io.BytesIO(face("a", (0.2, 0.3, 0.6))))
    blinking = Image.open(io.BytesIO(face("a", (0.2, 0.3, 0.6), eyes="closed")))
    night = Image.open(photos / "low-light.jpg")
    frames = [
        ("view-1.jpg", view.filter(ImageFilter.GaussianBlur(2.5)), "10:00:00"),
        ("view-2.jpg", view, "10:00:02"),
        ("view-3.jpg", view.filter(ImageFilter.GaussianBlur(1.5)), "10:00:03"),
        ("sitter-1.jpg", blinking, "11:30:00"),
        ("sitter-2.jpg", sitter, "11:30:04"),
        ("night.jpg", night, "21:00:00"),
    ]
    imported = import_bytes(
        client,
        *[
            (name, jpeg_with_exif(image=image, taken=f"2026:07:04 {time}"))
            for name, image, time in frames
        ],
    )
    return {p["filename"]: p["id"] for p in imported}


def cull(client: TestClient, **body: Any) -> dict[str, Any]:
    res = client.post("/api/library/cull", json=body)
    assert res.status_code == 200, res.text
    found: dict[str, Any] = res.json()
    return found


def names(result: dict[str, Any]) -> list[str]:
    return [pick["photo"]["filename"] for pick in result["picks"]]


def test_picks_the_best_shot_of_each_moment(client: TestClient, shoot: dict[str, str]) -> None:
    result = cull(client, count=3)
    assert result["source"] == "built-in"
    assert (result["considered"], result["moments"]) == (6, 3)
    assert sorted(names(result)) == ["night.jpg", "sitter-2.jpg", "view-2.jpg"]
    by_name = {pick["photo"]["filename"]: pick for pick in result["picks"]}
    assert by_name["view-2.jpg"]["moment_size"] == 3
    assert by_name["view-2.jpg"]["reasons"][:2] == ["Best of 3 similar shots", "sharpest of them"]
    assert "eyes open where others blinked" in by_name["sitter-2.jpg"]["reasons"]
    assert by_name["night.jpg"]["moment_size"] == 1
    assert result["summary"].startswith(
        "Picked 3 of 6 photos from 3 moments: the best shot of each"
    )
    assert "with closed eyes" in result["summary"]


def test_fewer_picks_take_the_strongest_moments(client: TestClient, shoot: dict[str, str]) -> None:
    result = cull(client, count=2)
    assert len(result["picks"]) == 2
    assert len({pick["moment"] for pick in result["picks"]}) == 2
    scores = [pick["score"] for pick in result["picks"]]
    assert scores == sorted(scores, reverse=True)


def test_more_picks_never_repeat_the_same_frame(client: TestClient, shoot: dict[str, str]) -> None:
    # The other frames of each moment are near copies of the pick, so there is nothing
    # more worth picking.
    assert len(cull(client, count=10)["picks"]) == 3


def test_what_the_person_said_counts(client: TestClient, shoot: dict[str, str]) -> None:
    client.post(
        "/api/library/photos/marks", json={"ids": [shoot["view-2.jpg"]], "flag": "rejected"}
    )
    picks = names(cull(client, count=3))
    assert "view-2.jpg" not in picks
    assert "view-3.jpg" in picks  # the next sharpest frame stands in
    client.post("/api/library/photos/marks", json={"ids": [shoot["night.jpg"]], "rating": 5})
    assert names(cull(client, count=1)) == ["night.jpg"]


def test_picks_from_a_selection(client: TestClient, shoot: dict[str, str]) -> None:
    ids = [shoot["view-1.jpg"], shoot["view-3.jpg"], shoot["night.jpg"]]
    assert sorted(names(cull(client, count=5, ids=ids))) == ["night.jpg", "view-3.jpg"]


@pytest.fixture
def advisor() -> Iterator[FakeAdvisor]:
    fake = FakeAdvisor()
    app.dependency_overrides[get_advisor] = lambda: fake
    yield fake
    app.dependency_overrides.pop(get_advisor, None)


def test_claude_explains_the_picks(
    client: TestClient, shoot: dict[str, str], advisor: FakeAdvisor
) -> None:
    advisor.answers.append(
        {
            "summary": "Three keepers, one from each part of the day.",
            "reasons": ["Crisp", "Nobody blinked", "Moody night light"],
        }
    )
    result = cull(client, count=3)
    assert result["source"] == "claude"
    assert result["summary"] == "Three keepers, one from each part of the day."
    assert [pick["reasons"] for pick in result["picks"]] == [
        ["Crisp"],
        ["Nobody blinked"],
        ["Moody night light"],
    ]
    call = advisor.calls[0]
    assert call["tier"] == "routine"
    text = call["content"][0]["text"]
    assert "view-2.jpg: sharpness" in text
    assert all(block["type"] == "text" for block in call["content"])  # no photos


@pytest.mark.parametrize(
    "answer",
    [AdvisorError("no"), {"summary": "Too few reasons.", "reasons": ["Crisp"]}],
)
def test_built_in_reasons_stand_when_claude_cannot_explain(
    client: TestClient, shoot: dict[str, str], advisor: FakeAdvisor, answer: Any
) -> None:
    advisor.answers.append(answer)
    result = cull(client, count=3)
    assert result["source"] == "built-in"
    assert result["summary"].startswith("Picked 3 of 6")


def test_searching_for_the_best_few(client: TestClient, shoot: dict[str, str]) -> None:
    res = client.post("/api/library/search", json={"query": "best 2 photos from july"})
    found = res.json()
    assert found["understood"]["best"] == 2
    assert len(found["photos"]) == 2
    assert found["picked"].startswith("Picked 2 of 6 photos")
