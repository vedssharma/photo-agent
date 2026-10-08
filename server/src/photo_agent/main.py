"""FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from photo_agent import __version__

app = FastAPI(title="photo-agent", version=__version__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class HealthResponse(BaseModel):
    status: str
    version: str


@app.get("/api/health", operation_id="getHealth", tags=["meta"])
def health() -> HealthResponse:
    """Report that the server is up."""
    return HealthResponse(status="ok", version=__version__)
