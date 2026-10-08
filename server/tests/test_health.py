from fastapi.testclient import TestClient

from photo_agent import __version__
from photo_agent.main import app

client = TestClient(app)


def test_health_reports_ok() -> None:
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok", "version": __version__}
