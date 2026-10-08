from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from photo_agent import __version__
from photo_agent.main import app
from photo_agent.settings import Settings, get_settings

client = TestClient(app)


@pytest.fixture
def settings() -> Iterator[Settings]:
    test_settings = Settings(_env_file=None, anthropic_api_key=None)
    app.dependency_overrides[get_settings] = lambda: test_settings
    yield test_settings
    app.dependency_overrides.clear()


def test_health_reports_ok(settings: Settings) -> None:
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json() == {
        "status": "ok",
        "version": __version__,
        "anthropic_configured": False,
    }


def test_health_reports_configured_key(settings: Settings) -> None:
    settings.anthropic_api_key = SecretStr("sk-ant-test")
    res = client.get("/api/health")
    assert res.json()["anthropic_configured"] is True
