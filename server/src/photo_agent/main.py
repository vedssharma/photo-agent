"""FastAPI application entry point."""

from typing import Annotated

from fastapi import Depends, FastAPI
from pydantic import BaseModel

from photo_agent import __version__
from photo_agent.settings import Settings, get_settings

app = FastAPI(title="photo-agent", version=__version__)


class HealthResponse(BaseModel):
    status: str
    version: str
    anthropic_configured: bool


@app.get("/api/health", operation_id="getHealth", tags=["meta"])
def health(settings: Annotated[Settings, Depends(get_settings)]) -> HealthResponse:
    """Report that the server is up and whether the Claude API key is set."""
    return HealthResponse(
        status="ok",
        version=__version__,
        anthropic_configured=settings.anthropic_api_key is not None,
    )
