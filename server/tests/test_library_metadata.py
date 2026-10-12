"""Photo details from EXIF, and filtering the library by date, camera, and place."""

from datetime import datetime
from pathlib import Path

from fastapi.testclient import TestClient
from library_helpers import import_bytes, jpeg_with_exif

from photo_agent.library import metadata


def test_reads_capture_time_camera_settings_and_location() -> None:
    found = metadata.read(
        jpeg_with_exif(
            taken="2025:07:14 18:30:05",
            make="Apple",
            model="iPhone 15 Pro",
            gps=(-33.8568, 151.2153),
            iso=125,
            exposure_time=1 / 250,
            f_number=1.8,
            focal_length=6.86,
            lens="iPhone 15 Pro back camera",
        )
    )
    assert found.taken_at == datetime(2025, 7, 14, 18, 30, 5)
    assert found.camera == "Apple iPhone 15 Pro"
    assert found.lens == "iPhone 15 Pro back camera"
    assert (found.iso, found.f_number, found.focal_length) == (125, 1.8, 6.86)
    assert found.exposure_time is not None and abs(found.exposure_time - 0.004) < 1e-6
    assert found.latitude is not None and abs(found.latitude + 33.8568) < 1e-3
    assert found.longitude is not None and abs(found.longitude - 151.2153) < 1e-3
    assert found.place == "-33.85,151.25"
    assert metadata.place_label(found.place) == "33.9° S, 151.2° E"


def test_missing_or_broken_metadata_is_left_empty(photos: Path) -> None:
    assert metadata.read((photos / "landscape.png").read_bytes()) == metadata.PhotoMetadata()
    assert metadata.read(b"not an image") == metadata.PhotoMetadata()
    odd = metadata.read(jpeg_with_exif(taken="0000:00:00 00:00:00", make="  Canon\x00 "))
    assert odd.taken_at is None
    assert odd.camera == "Canon"


def test_camera_names_do_not_repeat_the_make() -> None:
    assert metadata.camera_name("Canon", "Canon EOS R5") == "Canon EOS R5"
    assert metadata.camera_name("FUJIFILM", "X-T5") == "FUJIFILM X-T5"
    assert metadata.camera_name(None, "X-T5") == "X-T5"
    assert metadata.camera_name("Sony", None) == "Sony"


def library(client: TestClient) -> dict[str, str]:
    imported = import_bytes(
        client,
        (
            "beach.jpg",
            jpeg_with_exif(
                taken="2025:07:14 10:00:00",
                make="Apple",
                model="iPhone 15 Pro",
                gps=(36.97, -122.03),
                color=(200, 180, 120),
            ),
        ),
        (
            "sunset.jpg",
            jpeg_with_exif(
                taken="2025:07:15 20:10:00",
                make="Apple",
                model="iPhone 15 Pro",
                gps=(36.96, -122.02),
                color=(220, 120, 60),
            ),
        ),
        (
            "snow.jpg",
            jpeg_with_exif(
                taken="2024:12:24 09:00:00", make="FUJIFILM", model="X-T5", color=(230, 235, 240)
            ),
        ),
        ("scan.jpg", jpeg_with_exif(color=(90, 90, 90))),
    )
    return {p["filename"]: p["id"] for p in imported}


def names(client: TestClient, **params: str | bool) -> list[str]:
    res = client.get("/api/library/photos", params=params)
    assert res.status_code == 200, res.text
    return [p["filename"] for p in res.json()]


def test_photos_carry_their_details_and_sort_by_capture_time(client: TestClient) -> None:
    ids = library(client)
    photo = client.get(f"/api/library/photos/{ids['beach.jpg']}").json()
    assert photo["taken_at"] == "2025-07-14T10:00:00"
    assert photo["camera"] == "Apple iPhone 15 Pro"
    assert photo["place"] == "36.95,-122.05"
    # Newest capture first; the scan has no date, so it goes by its import time.
    assert names(client)[1:] == ["sunset.jpg", "beach.jpg", "snow.jpg"]
    assert names(client)[0] == "scan.jpg"


def test_filters_by_date_camera_and_place(client: TestClient) -> None:
    library(client)
    assert names(client, taken_after="2025-01-01") == ["sunset.jpg", "beach.jpg"]
    assert names(client, taken_before="2025-07-14") == ["beach.jpg", "snow.jpg"]
    assert names(client, taken_after="2025-07-15", taken_before="2025-07-15") == ["sunset.jpg"]
    assert names(client, camera="FUJIFILM X-T5") == ["snow.jpg"]
    assert names(client, place="36.95,-122.05") == ["sunset.jpg", "beach.jpg"]
    assert names(client, has_location=False) == ["scan.jpg", "snow.jpg"]
    assert names(client, camera="Apple iPhone 15 Pro", taken_before="2025-07-14") == ["beach.jpg"]


def test_facets_list_what_there_is_to_filter_by(client: TestClient) -> None:
    library(client)
    facets = client.get("/api/library/facets").json()
    assert facets["cameras"] == [
        {"value": "Apple iPhone 15 Pro", "label": "Apple iPhone 15 Pro", "count": 2},
        {"value": "FUJIFILM X-T5", "label": "FUJIFILM X-T5", "count": 1},
    ]
    assert facets["places"] == [
        {"value": "36.95,-122.05", "label": "37.0° N, 122.0° W", "count": 2}
    ]
    assert [(y["value"], y["count"]) for y in facets["years"]] == [("2025", 2), ("2024", 1)]
