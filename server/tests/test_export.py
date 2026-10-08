import io
from collections.abc import Callable
from typing import Any

from fastapi.testclient import TestClient
from PIL import Image

from photo_agent.export import GPS_IFD
from photo_agent.graph import Step
from photo_agent.operations import Crop, Exposure
from photo_agent.routes import get_store
from photo_agent.settings import Settings

Upload = Callable[[str], dict[str, Any]]
EXIF_ORIENTATION = 0x0112


def apply(settings: Settings, doc_id: str, *ops: Any) -> None:
    store = get_store(settings)
    doc = store.get(doc_id)
    doc.commit(Step(label="test", operations=list(ops)))
    store.save(doc)


def test_export_jpeg_at_full_resolution(
    client: TestClient, upload: Upload, settings: Settings
) -> None:
    doc = upload("phone.heic")
    apply(settings, doc["id"], Crop(aspect="1:1"), Exposure(stops=0.3))
    res = client.post(f"/api/documents/{doc['id']}/export", json={"quality": 80})
    assert res.status_code == 200
    assert res.headers["content-type"] == "image/jpeg"
    assert "filename*=UTF-8''phone-edited.jpg" in res.headers["content-disposition"]
    img = Image.open(io.BytesIO(res.content))
    assert img.format == "JPEG"
    assert img.size == (300, 300)
    assert img.info.get("icc_profile")


def test_export_strips_location_by_default(client: TestClient, upload: Upload) -> None:
    doc = upload("phone.heic")
    res = client.post(f"/api/documents/{doc['id']}/export", json={})
    exif = Image.open(io.BytesIO(res.content)).getexif()
    assert GPS_IFD not in exif
    assert exif  # other camera metadata survives

    res = client.post(f"/api/documents/{doc['id']}/export", json={"keep_location": True})
    assert GPS_IFD in Image.open(io.BytesIO(res.content)).getexif()


def test_export_resets_orientation_of_upright_pixels(client: TestClient, upload: Upload) -> None:
    doc = upload("portrait-exif-rotated.jpg")
    res = client.post(f"/api/documents/{doc['id']}/export", json={"format": "png"})
    img = Image.open(io.BytesIO(res.content))
    assert img.format == "PNG"
    assert img.getexif().get(EXIF_ORIENTATION, 1) == 1
    assert res.headers["content-type"] == "image/png"


def test_export_validates_options(client: TestClient, upload: Upload) -> None:
    doc = upload("portrait.jpg")
    res = client.post(f"/api/documents/{doc['id']}/export", json={"format": "gif"})
    assert res.status_code == 422
