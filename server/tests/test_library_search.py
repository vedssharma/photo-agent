"""Auto-tagging and natural-language search."""

from collections.abc import Iterator
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fake_advisor import FakeAdvisor
from fastapi.testclient import TestClient
from library_helpers import BEACH, NIGHT, SNOW, import_bytes, scene

from photo_agent.advisor import AdvisorError
from photo_agent.library import dates
from photo_agent.library.search import understand_built_in
from photo_agent.library.vocabulary import tags_in
from photo_agent.main import app
from photo_agent.routes import get_advisor, get_library
from photo_agent.settings import Settings
from photo_agent.vision.worker import JobResult

TODAY = date(2026, 10, 12)


def span(text: str, today: date = TODAY) -> tuple[date, date] | None:
    return dates.find_dates(text, today)[0]


def test_reads_dates_in_everyday_words() -> None:
    assert span("beach photos from last summer") == (date(2026, 6, 1), date(2026, 8, 31))
    assert span("last summer", date(2026, 7, 10)) == (date(2025, 6, 1), date(2025, 8, 31))
    assert span("this summer", date(2026, 7, 10)) == (date(2026, 6, 1), date(2026, 8, 31))
    assert span("summer of 2024") == (date(2024, 6, 1), date(2024, 8, 31))
    assert span("last winter") == (date(2025, 12, 1), date(2026, 2, 28))
    assert span("winter 2024") == (date(2024, 12, 1), date(2025, 2, 28))
    assert span("in 2023") == (date(2023, 1, 1), date(2023, 12, 31))
    assert span("last month") == (date(2026, 9, 1), date(2026, 9, 30))
    assert span("this year") == (date(2026, 1, 1), date(2026, 12, 31))
    assert span("yesterday") == (date(2026, 10, 11), date(2026, 10, 11))
    assert span("last week") == (date(2026, 10, 5), date(2026, 10, 11))
    assert span("in July") == (date(2026, 7, 1), date(2026, 7, 31))
    assert span("in december") == (date(2025, 12, 1), date(2025, 12, 31))
    assert span("march 2025") == (date(2025, 3, 1), date(2025, 3, 31))
    assert span("photos that may be blurry") is None
    assert dates.find_dates("dog last summer at the lake", TODAY)[1] == "dog at the lake"


def test_maps_everyday_words_onto_tags() -> None:
    found, rest = tags_in("Puppy at the seaside in black and white, red umbrella")
    assert found == ["black and white", "beach", "dog"]
    assert rest == "at the in , red umbrella"


def test_built_in_understanding() -> None:
    u = understand_built_in("beach photos from last summer with the dog", TODAY, [])
    assert (u.tags, u.taken_after, u.taken_before, u.words) == (
        ["beach", "dog"],
        date(2026, 6, 1),
        date(2026, 8, 31),
        "",
    )
    u = understand_built_in("the best 30 from the Lisbon trip", TODAY, [])
    assert (u.best, u.words) == (30, "lisbon trip")
    u = understand_built_in("5 star favorites shot on FUJIFILM X-T5", TODAY, ["FUJIFILM X-T5"])
    assert (u.min_rating, u.flag, u.camera, u.words) == (5, "picked", "FUJIFILM X-T5", "")
    assert understand_built_in("best sunsets", TODAY, []).best == 30


@pytest.fixture
def tagged(client: TestClient) -> dict[str, str]:
    from library_helpers import jpeg_with_exif

    imported = import_bytes(
        client,
        ("beach.jpg", scene(*BEACH)),
        ("snow.jpg", scene(*SNOW)),
        ("night.jpg", scene(*NIGHT)),
        ("IMG_2041.jpg", jpeg_with_exif(taken="2026:07:04 12:00:00", color=(128, 128, 128))),
    )
    return {p["filename"]: p["id"] for p in imported}


def photo(client: TestClient, photo_id: str) -> dict[str, Any]:
    body: dict[str, Any] = client.get(f"/api/library/photos/{photo_id}").json()
    return body


def test_photos_are_tagged_when_indexed(client: TestClient, tagged: dict[str, str]) -> None:
    assert "beach" in photo(client, tagged["beach.jpg"])["tags"]
    assert "snow" in photo(client, tagged["snow.jpg"])["tags"]
    assert "night" in photo(client, tagged["night.jpg"])["tags"]
    counts = {t["tag"]: t["count"] for t in client.get("/api/library/tags").json()}
    assert counts["beach"] == 1


def test_tags_by_hand_and_filtering_by_tag(client: TestClient, tagged: dict[str, str]) -> None:
    beach, snow = tagged["beach.jpg"], tagged["snow.jpg"]
    res = client.post("/api/library/photos/tags", json={"ids": [beach, snow], "add": [" Trip "]})
    assert all("trip" in p["tags"] for p in res.json())
    client.post("/api/library/photos/tags", json={"ids": [beach], "remove": ["beach"]})
    assert "beach" not in photo(client, beach)["tags"]

    def names(**q: list[str]) -> list[str]:
        return sorted(p["filename"] for p in client.get("/api/library/photos", params=q).json())

    assert names(tags=["trip"]) == ["beach.jpg", "snow.jpg"]
    assert names(tags=["trip", "snow"]) == ["snow.jpg"]
    assert names(tags=["beach"]) == []


def test_a_removed_tag_stays_removed_when_tagged_again(
    client: TestClient, settings: Settings, tagged: dict[str, str]
) -> None:
    beach = tagged["beach.jpg"]
    client.post("/api/library/photos/tags", json={"ids": [beach], "remove": ["beach"]})
    library = get_library(settings)
    from photo_agent.library import tagging

    tagging.tag_photo(library, beach)
    assert "beach" not in photo(client, beach)["tags"]


def search(client: TestClient, query: str, **extra: Any) -> dict[str, Any]:
    res = client.post("/api/library/search", json={"query": query, **extra})
    assert res.status_code == 200, res.text
    body: dict[str, Any] = res.json()
    return body


def test_search_with_built_in_rules(client: TestClient, tagged: dict[str, str]) -> None:
    found = search(client, "snowy photos")
    assert found["source"] == "built-in"
    assert [p["filename"] for p in found["photos"]] == ["snow.jpg"]

    found = search(client, "photos from July 2026")
    assert found["understood"]["taken_after"] == "2026-07-01"
    assert [p["filename"] for p in found["photos"]] == ["IMG_2041.jpg"]
    # Left-over words without a vision model match file names.
    found = search(client, "IMG_2041")
    assert [p["filename"] for p in found["photos"]] == ["IMG_2041.jpg"]
    assert found["matched_words_by"] == "file name"
    assert search(client, "red umbrella")["photos"] == []


@pytest.fixture
def advisor() -> Iterator[FakeAdvisor]:
    fake = FakeAdvisor()
    app.dependency_overrides[get_advisor] = lambda: fake
    yield fake
    app.dependency_overrides.pop(get_advisor, None)


def test_search_understood_by_claude(
    client: TestClient, tagged: dict[str, str], advisor: FakeAdvisor
) -> None:
    advisor.answers.append(
        {
            "tags": ["snow", "unicorn"],
            "taken_after": None,
            "taken_before": None,
            "camera": "Nonexistent",
            "min_rating": 0,
            "flag": None,
            "best": 10,
            "words": "",
        }
    )
    found = search(client, "my 10 best winter wonderland shots")
    assert found["source"] == "claude"
    assert found["understood"]["tags"] == ["snow"]
    assert found["understood"]["camera"] is None
    assert found["understood"]["best"] == 10
    assert found["understood"]["words"] == "unicorn"
    assert found["summary"].startswith("The best 10: tagged snow")
    call = advisor.calls[0]
    assert call["tier"] == "routine"
    assert "winter wonderland" in call["content"][0]["text"]
    # Only words go to Claude: no photos.
    assert all(block["type"] == "text" for block in call["content"])


def test_search_falls_back_when_claude_fails(
    client: TestClient, tagged: dict[str, str], advisor: FakeAdvisor
) -> None:
    advisor.answers.append(AdvisorError("no"))
    found = search(client, "night")
    assert found["source"] == "built-in"
    assert [p["filename"] for p in found["photos"]] == ["night.jpg"]


class LookingWorker:
    """Stands in for the vision model: embeds text as a fixed direction."""

    def run(self, task: str, image: Any = None, params: Any = None, **_: Any) -> JobResult:
        assert task == "embed_text"
        return JobResult({"vectors": [[1.0, 0.0]], "scale": 10.0, "bias": -5.0}, "fake-model")


def test_left_over_words_are_matched_by_look(
    client: TestClient, settings: Settings, tagged: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    library = get_library(settings)
    vectors = {"beach.jpg": [0.9, 0.1], "snow.jpg": [0.7, 0.7], "night.jpg": [0.0, 1.0]}
    for name, vector in vectors.items():
        library.db.run(
            "INSERT OR REPLACE INTO embeddings (photo_id, model, vector) VALUES (?, ?, ?)",
            (tagged[name], "fake-model", np.asarray(vector, np.float32).tobytes()),
        )
    monkeypatch.setattr(library, "worker", LookingWorker())
    found = search(client, "red umbrella")
    assert found["matched_words_by"] == "look"
    assert [p["filename"] for p in found["photos"]] == ["beach.jpg", "snow.jpg"]


def test_search_within_an_album(client: TestClient, tagged: dict[str, str], photos: Path) -> None:
    album = client.post(
        "/api/library/albums", json={"name": "Cold", "photo_ids": [tagged["night.jpg"]]}
    ).json()
    assert search(client, "snow", album=album["id"])["photos"] == []
