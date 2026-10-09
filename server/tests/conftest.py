from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from photo_agent.main import app
from photo_agent.settings import REPO_ROOT, Settings, get_settings

PHOTOS = REPO_ROOT / "fixtures" / "photos"


@pytest.fixture
def photos() -> Path:
    """Directory holding the shared test photos (see fixtures/README.md)."""
    return PHOTOS


@pytest.fixture
def settings(tmp_path: Path) -> Iterator[Settings]:
    """App settings with no API key, a throwaway data directory, and models replaced by
    their classical fallbacks on a thread."""
    test_settings = Settings(
        _env_file=None,
        anthropic_api_key=None,
        data_dir=tmp_path / "data",
        model_worker="inline",
        model_backends="classical",
    )
    app.dependency_overrides[get_settings] = lambda: test_settings
    yield test_settings
    app.dependency_overrides.clear()


@pytest.fixture
def client(settings: Settings) -> TestClient:
    return TestClient(app)


@pytest.fixture
def upload(client: TestClient, photos: Path) -> Callable[[str], dict[str, Any]]:
    """Upload a fixture photo and return the new document."""

    def _upload(name: str) -> dict[str, Any]:
        res = client.post("/api/documents", files={"file": (name, (photos / name).read_bytes())})
        assert res.status_code == 201, res.text
        body: dict[str, Any] = res.json()
        return body

    return _upload
