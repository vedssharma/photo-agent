from pathlib import Path

import pytest

from photo_agent.settings import REPO_ROOT

PHOTOS = REPO_ROOT / "fixtures" / "photos"


@pytest.fixture
def photos() -> Path:
    """Directory holding the shared test photos (see fixtures/README.md)."""
    return PHOTOS
