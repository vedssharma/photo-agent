"""Finding duplicates and near duplicates."""

import io
from pathlib import Path
from typing import Any

import numpy as np
from fastapi.testclient import TestClient
from library_helpers import import_bytes
from PIL import Image, ImageEnhance

from photo_agent.library.fingerprints import fingerprint


def jpeg(image: Image.Image, quality: int = 80) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def variants(photos: Path) -> dict[str, bytes]:
    """The landscape fixture as it might turn up again, and two other photos."""
    base = Image.open(photos / "landscape.png").convert("RGB")
    w, h = base.size
    return {
        "landscape.png": (photos / "landscape.png").read_bytes(),
        "small copy.jpg": jpeg(base.resize((w // 3, h // 3))),
        "brighter.jpg": jpeg(ImageEnhance.Brightness(base).enhance(1.2)),
        "cropped.jpg": jpeg(base.crop((int(w * 0.05), int(h * 0.03), w, h))),
        "portrait.jpg": (photos / "portrait.jpg").read_bytes(),
        "low-light.jpg": (photos / "low-light.jpg").read_bytes(),
    }


def bits_apart(a: str, b: str) -> int:
    return (int(a, 16) ^ int(b, 16)).bit_count()


def test_fingerprints_survive_resizing_but_tell_photos_apart(photos: Path) -> None:
    def fp(data: bytes) -> str:
        rgb = Image.open(io.BytesIO(data)).convert("RGB")
        return fingerprint((np.asarray(rgb, np.float32) / 255).astype(np.float32))

    found = {name: fp(data) for name, data in variants(photos).items()}
    assert len(found["landscape.png"]) == 16
    assert bits_apart(found["landscape.png"], found["small copy.jpg"]) <= 2
    assert bits_apart(found["landscape.png"], found["cropped.jpg"]) <= 10
    assert bits_apart(found["landscape.png"], found["portrait.jpg"]) > 20


def duplicates(client: TestClient) -> list[dict[str, Any]]:
    res = client.get("/api/library/duplicates")
    assert res.status_code == 200
    found: list[dict[str, Any]] = res.json()
    return found


def names(group: dict[str, Any]) -> list[str]:
    return [p["filename"] for p in group["photos"]]


def test_groups_copies_of_a_photo_and_suggests_one_to_keep(
    client: TestClient, photos: Path
) -> None:
    imported = import_bytes(client, *variants(photos).items())
    (group,) = duplicates(client)
    assert sorted(names(group)) == [
        "brighter.jpg",
        "cropped.jpg",
        "landscape.png",
        "small copy.jpg",
    ]
    assert group["kind"] == "near"  # the crop is not quite the same photo
    # The biggest is suggested, unless one has been picked, rated, or edited.
    assert names(group)[0] in ("landscape.png", "brighter.jpg")
    by_name = {p["filename"]: p["id"] for p in imported}
    assert group["keep"] == by_name[names(group)[0]]

    client.post("/api/library/photos/marks", json={"ids": [by_name["small copy.jpg"]], "rating": 4})
    (group,) = duplicates(client)
    assert group["keep"] == by_name["small copy.jpg"]


def test_only_resized_copies_are_exact(client: TestClient, photos: Path) -> None:
    found = variants(photos)
    import_bytes(
        client,
        ("landscape.png", found["landscape.png"]),
        ("small copy.jpg", found["small copy.jpg"]),
    )
    (group,) = duplicates(client)
    assert group["kind"] == "exact"


def test_photos_said_to_differ_are_not_grouped_again(client: TestClient, photos: Path) -> None:
    found = variants(photos)
    imported = import_bytes(
        client,
        ("landscape.png", found["landscape.png"]),
        ("cropped.jpg", found["cropped.jpg"]),
        ("portrait.jpg", found["portrait.jpg"]),
    )
    assert len(duplicates(client)) == 1
    ids = [p["id"] for p in imported[:2]]
    res = client.post("/api/library/duplicates/dismiss", json={"ids": ids})
    assert res.status_code == 204
    assert duplicates(client) == []


def test_removing_the_extras(client: TestClient, photos: Path) -> None:
    import_bytes(client, *variants(photos).items())
    (group,) = duplicates(client)
    extras = [p["id"] for p in group["photos"] if p["id"] != group["keep"]]
    client.post("/api/library/photos/remove", json={"ids": extras})
    assert duplicates(client) == []
    assert len(client.get("/api/library/photos").json()) == 3
