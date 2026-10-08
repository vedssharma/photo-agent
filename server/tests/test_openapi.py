import json

from photo_agent.main import app
from photo_agent.openapi import DEFAULT_PATH


def test_committed_openapi_matches_app() -> None:
    """api/openapi.json must be regenerated (`make api`) whenever routes change."""
    committed = json.loads(DEFAULT_PATH.read_text())
    assert committed == app.openapi()
