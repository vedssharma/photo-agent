"""People albums: finding faces, grouping them into people, and correcting the groups."""

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from fastapi.testclient import TestClient
from library_helpers import face, import_bytes

from photo_agent.vision import library as classical_library

BLUE, GREEN, GRAY = (0.2, 0.3, 0.6), (0.3, 0.5, 0.3), (0.6, 0.6, 0.6)


def turn_on(client: TestClient, on: bool = True) -> None:
    res = client.put("/api/library/settings", json={"faces": on})
    assert res.status_code == 200, res.text
    assert res.json() == {"faces": on}


@pytest.fixture
def shoot(client: TestClient) -> dict[str, str]:
    """Four photos: person a twice, person b twice, each against different backgrounds."""
    imported = import_bytes(
        client,
        ("a1.png", face("a", BLUE)),
        ("a2.png", face("a", GREEN, dx=30, seed=1)),
        ("b1.png", face("b", BLUE, seed=2)),
        ("b2.png", face("b", GRAY, dx=-20, seed=3)),
    )
    return {p["filename"].removesuffix(".png"): p["id"] for p in imported}


def people(client: TestClient) -> list[dict[str, Any]]:
    res = client.get("/api/library/people")
    assert res.status_code == 200
    found: list[dict[str, Any]] = res.json()
    return found


def faces_in(client: TestClient, photo_id: str) -> list[dict[str, Any]]:
    res = client.get(f"/api/library/photos/{photo_id}/faces")
    assert res.status_code == 200
    found: list[dict[str, Any]] = res.json()
    return found


def person_of(client: TestClient, photo_id: str) -> str:
    (only,) = faces_in(client, photo_id)
    person: str = only["person"]
    return person


def test_classical_faces_tell_the_drawn_people_apart() -> None:
    import io

    from PIL import Image

    def vector(data: bytes) -> npt.NDArray[np.float64]:
        rgb = Image.open(io.BytesIO(data)).convert("RGB")
        pixels = (np.asarray(rgb, np.float32) / 255).astype(np.float32)
        (found,) = classical_library.faces(pixels)
        assert all(0 <= v <= 1 for v in found["box"])
        return np.asarray(found["embedding"])

    a1, a2 = vector(face("a", BLUE)), vector(face("a", GREEN, dx=30, seed=1))
    b1 = vector(face("b", BLUE, seed=2))
    assert a1 @ a2 > 0.9
    assert a1 @ b1 < 0.7


def test_people_albums_are_off_until_turned_on(client: TestClient, shoot: dict[str, str]) -> None:
    assert client.get("/api/library/settings").json() == {"faces": False}
    assert people(client) == []
    assert faces_in(client, shoot["a1"]) == []


def test_faces_are_grouped_into_people(client: TestClient, shoot: dict[str, str]) -> None:
    turn_on(client)
    found = people(client)
    assert len(found) == 2
    assert all(p["name"] is None and p["photos"] == 2 for p in found)
    assert person_of(client, shoot["a1"]) == person_of(client, shoot["a2"])
    assert person_of(client, shoot["b1"]) == person_of(client, shoot["b2"])
    assert person_of(client, shoot["a1"]) != person_of(client, shoot["b1"])
    photo = client.get(f"/api/library/photos/{shoot['a1']}").json()
    assert photo["people"] == [person_of(client, shoot["a1"])]
    thumb = client.get(f"/api/library/faces/{found[0]['cover']}/thumbnail")
    assert thumb.status_code == 200
    assert thumb.headers["content-type"] == "image/jpeg"


def test_new_photos_join_the_people_already_found(
    client: TestClient, shoot: dict[str, str]
) -> None:
    turn_on(client)
    (a3,) = import_bytes(client, ("a3.png", face("a", GRAY, dx=-25, seed=4)))
    assert person_of(client, a3["id"]) == person_of(client, shoot["a1"])
    assert sorted(p["photos"] for p in people(client)) == [2, 3]


def test_naming_and_filtering_and_searching_by_person(
    client: TestClient, shoot: dict[str, str]
) -> None:
    turn_on(client)
    maya = person_of(client, shoot["a1"])
    res = client.patch(f"/api/library/people/{maya}", json={"name": "  Maya "})
    assert res.json()["name"] == "Maya"
    assert people(client)[0]["id"] == maya  # named people first
    assert faces_in(client, shoot["a2"])[0]["person_name"] == "Maya"

    listed = client.get("/api/library/photos", params={"people": [maya]}).json()
    assert sorted(p["filename"] for p in listed) == ["a1.png", "a2.png"]

    found = client.post("/api/library/search", json={"query": "photos with Maya"}).json()
    assert found["understood"]["people"] == ["Maya"]
    assert found["summary"] == "With Maya"
    assert sorted(p["filename"] for p in found["photos"]) == ["a1.png", "a2.png"]


def test_merging_two_people(client: TestClient, shoot: dict[str, str]) -> None:
    turn_on(client)
    a, b = person_of(client, shoot["a1"]), person_of(client, shoot["b1"])
    res = client.post(f"/api/library/people/{b}/merge", json={"into": a})
    assert res.status_code == 200
    assert res.json()["photos"] == 4
    assert [p["id"] for p in people(client)] == [a]
    assert client.post("/api/library/people/nope/merge", json={"into": a}).status_code == 404


def test_not_this_person_sticks(client: TestClient, shoot: dict[str, str]) -> None:
    turn_on(client)
    a = person_of(client, shoot["a1"])
    (wrong,) = faces_in(client, shoot["a2"])
    res = client.post(f"/api/library/faces/{wrong['id']}", json={"person": None})
    assert res.json()["person"] is None
    # The person had only two faces; with one left they are still there.
    assert {p["id"]: p["photos"] for p in people(client)}[a] == 1
    # A new photo of a still joins a, but the face taken away is never put back.
    import_bytes(client, ("a3.png", face("a", GRAY, dx=-25, seed=4)))
    assert faces_in(client, shoot["a2"])[0]["person"] is None


def test_moving_a_face_to_someone_else(client: TestClient, shoot: dict[str, str]) -> None:
    turn_on(client)
    a, b = person_of(client, shoot["a1"]), person_of(client, shoot["b1"])
    (face_b2,) = faces_in(client, shoot["b2"])
    client.post(f"/api/library/faces/{face_b2['id']}", json={"person": a})
    assert person_of(client, shoot["b2"]) == a
    assert {p["id"]: p["photos"] for p in people(client)} == {a: 3, b: 1}
    res = client.post(f"/api/library/faces/{face_b2['id']}", json={"person": "nobody"})
    assert res.status_code == 404


def test_forgetting_a_person(client: TestClient, shoot: dict[str, str]) -> None:
    turn_on(client)
    a = person_of(client, shoot["a1"])
    assert client.delete(f"/api/library/people/{a}").status_code == 204
    assert [p["photos"] for p in people(client)] == [2]
    assert faces_in(client, shoot["a1"])[0]["person"] is None
    # Their faces are left alone from now on, even when another photo of them comes in.
    import_bytes(client, ("a3.png", face("a", GRAY, dx=-25, seed=4)))
    assert len(people(client)) == 1
    assert client.delete(f"/api/library/people/{a}").status_code == 404


def test_turning_it_off_forgets_every_face(client: TestClient, shoot: dict[str, str]) -> None:
    turn_on(client)
    assert len(people(client)) == 2
    turn_on(client, False)
    assert people(client) == []
    assert faces_in(client, shoot["a1"]) == []
    assert client.get("/api/library/status").json()["pending"] == 0
    # On again: found afresh.
    turn_on(client)
    assert len(people(client)) == 2
