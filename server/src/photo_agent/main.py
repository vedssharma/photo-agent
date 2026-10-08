"""FastAPI application entry point."""

from typing import Annotated, Any

from fastapi import Depends, FastAPI
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, TypeAdapter

from photo_agent import __version__, routes
from photo_agent.agent import ChatEvent, ChatMessage
from photo_agent.controls import OperationSpec, operation_specs
from photo_agent.settings import Settings, get_settings

app = FastAPI(title="photo-agent", version=__version__)
app.include_router(routes.router)


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


@app.get("/api/operations", operation_id="listOperations", tags=["meta"])
def operations() -> list[OperationSpec]:
    """Every operation with its parameters' ranges, for building manual controls."""
    return operation_specs()


def openapi() -> dict[str, Any]:
    """OpenAPI schema, plus the chat WebSocket's message types (OpenAPI cannot describe
    WebSockets, but the web client still wants generated types for them)."""
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(title=app.title, version=app.version, routes=app.routes)
    components = schema.setdefault("components", {}).setdefault("schemas", {})
    ref = "#/components/schemas/{model}"
    for name, adapter, mode in (
        ("ChatEvent", TypeAdapter(ChatEvent), "serialization"),
        ("ChatMessage", TypeAdapter(ChatMessage), "validation"),
    ):
        extra = adapter.json_schema(ref_template=ref, mode=mode)  # type: ignore[arg-type]
        for def_name, definition in extra.pop("$defs", {}).items():
            components.setdefault(def_name, definition)
        components.setdefault(name, extra)
    app.openapi_schema = schema
    return schema


app.openapi = openapi  # type: ignore[method-assign]
