"""Builders shared by the library tests."""

from pathlib import Path
from typing import Any

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

PHOTOS = ["landscape.png", "portrait.jpg", "low-light.jpg"]


def files(photos: Path, *names: str) -> list[tuple[str, tuple[str, bytes]]]:
    return [("files", (name, (photos / name).read_bytes())) for name in names]


def mean(image: Image.Image) -> float:
    return float(np.asarray(image.convert("L"), np.float32).mean())


def import_photos(client: TestClient, photos: Path, *names: str) -> dict[str, Any]:
    res = client.post("/api/library/photos", files=files(photos, *names))
    assert res.status_code == 201, res.text
    body: dict[str, Any] = res.json()
    return body
