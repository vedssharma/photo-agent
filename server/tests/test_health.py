from fastapi.testclient import TestClient
from pydantic import SecretStr

from photo_agent import __version__
from photo_agent.settings import Settings


def test_health_reports_ok(client: TestClient) -> None:
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json() == {
        "status": "ok",
        "version": __version__,
        "anthropic_configured": False,
    }


def test_health_reports_configured_key(client: TestClient, settings: Settings) -> None:
    settings.anthropic_api_key = SecretStr("sk-ant-test")
    res = client.get("/api/health")
    assert res.json()["anthropic_configured"] is True
