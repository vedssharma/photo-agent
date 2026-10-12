"""Organizing the library: ratings, flags, color labels, and albums."""

from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from library_helpers import PHOTOS, import_photos


def library(client: TestClient, photos: Path) -> dict[str, str]:
    imported = import_photos(client, photos, *PHOTOS)["imported"]
    return {p["filename"]: p["id"] for p in imported}


def listed(client: TestClient, **params: Any) -> list[str]:
    res = client.get("/api/library/photos", params=params)
    assert res.status_code == 200, res.text
    return sorted(p["filename"] for p in res.json())


def test_rate_flag_and_label_several_photos(client: TestClient, photos: Path) -> None:
    ids = library(client, photos)
    both = [ids["portrait.jpg"], ids["landscape.png"]]
    res = client.post(
        "/api/library/photos/marks", json={"ids": both, "rating": 4, "flag": "picked"}
    )
    body = res.json()
    assert body["changed"] == 2
    assert {(p["rating"], p["flag"], p["label"]) for p in body["photos"]} == {(4, "picked", None)}

    client.post(
        "/api/library/photos/marks",
        json={"ids": [ids["portrait.jpg"]], "rating": 2, "label": "red"},
    )
    client.post(
        "/api/library/photos/marks", json={"ids": [ids["low-light.jpg"]], "flag": "rejected"}
    )

    assert listed(client, min_rating=3) == ["landscape.png"]
    assert listed(client, min_rating=2) == ["landscape.png", "portrait.jpg"]
    assert listed(client, flag="picked") == ["landscape.png", "portrait.jpg"]
    assert listed(client, flag="rejected") == ["low-light.jpg"]
    assert listed(client, label="red") == ["portrait.jpg"]
    assert listed(client, label="red", flag="picked", min_rating=1) == ["portrait.jpg"]

    # Clearing a label, and what is left alone.
    client.post("/api/library/photos/marks", json={"ids": [ids["portrait.jpg"]], "label": "none"})
    photo = client.get(f"/api/library/photos/{ids['portrait.jpg']}").json()
    assert (photo["rating"], photo["flag"], photo["label"]) == (2, "picked", None)


def test_marks_are_validated(client: TestClient, photos: Path) -> None:
    ids = library(client, photos)
    one = [ids["portrait.jpg"]]
    assert (
        client.post("/api/library/photos/marks", json={"ids": one, "rating": 6}).status_code == 422
    )
    assert (
        client.post("/api/library/photos/marks", json={"ids": one, "label": "pink"}).status_code
        == 422
    )
    assert client.post("/api/library/photos/marks", json={"ids": []}).status_code == 422


def test_albums(client: TestClient, photos: Path) -> None:
    ids = library(client, photos)
    trip = client.post(
        "/api/library/albums",
        json={"name": "  Trip ", "photo_ids": [ids["landscape.png"], ids["portrait.jpg"]]},
    )
    assert trip.status_code == 201
    album = trip.json()
    assert (album["name"], album["count"]) == ("Trip", 2)
    assert album["cover"] in (ids["landscape.png"], ids["portrait.jpg"])

    assert listed(client, album=album["id"]) == ["landscape.png", "portrait.jpg"]
    photo = client.get(f"/api/library/photos/{ids['portrait.jpg']}").json()
    assert photo["albums"] == [album["id"]]

    url = f"/api/library/albums/{album['id']}"
    assert (
        client.post(
            f"{url}/photos", json={"ids": [ids["low-light.jpg"], ids["portrait.jpg"]]}
        ).json()["count"]
        == 3
    )
    assert (
        client.post(f"{url}/photos/remove", json={"ids": [ids["portrait.jpg"]]}).json()["count"]
        == 2
    )
    assert listed(client, album=album["id"]) == ["landscape.png", "low-light.jpg"]

    client.post("/api/library/albums", json={"name": "after"})
    assert client.patch(url, json={"name": "Beach trip"}).json()["name"] == "Beach trip"
    assert [a["name"] for a in client.get("/api/library/albums").json()] == ["after", "Beach trip"]

    # Deleting a photo takes it out of its albums; deleting an album keeps its photos.
    client.post("/api/library/photos/remove", json={"ids": [ids["landscape.png"]]})
    assert client.get("/api/library/albums").json()[1]["count"] == 1
    assert client.delete(url).status_code == 204
    assert len(listed(client)) == 2
    assert client.delete(url).status_code == 404
    assert client.post(f"{url}/photos", json={"ids": [ids["portrait.jpg"]]}).status_code == 404
    assert client.patch(url, json={"name": "x"}).status_code == 404
