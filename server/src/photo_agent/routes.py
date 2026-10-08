"""HTTP routes for documents: upload, inspect, undo/redo, previews, chat, and export."""

from __future__ import annotations

import asyncio
from functools import lru_cache
from pathlib import Path
from typing import Annotated
from urllib.parse import quote

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.responses import Response

from photo_agent import imaging
from photo_agent.agent import AgentError, AgentEvent, AgentService, ClaudeModel, ModelClient
from photo_agent.export import ExportOptions, export_bytes, export_filename
from photo_agent.graph import Document, DocumentView
from photo_agent.render import RenderCache
from photo_agent.settings import Settings, get_settings
from photo_agent.store import DocumentNotFoundError, DocumentStore

MAX_UPLOAD_BYTES = 64 * 1024 * 1024
PREVIEW_QUALITY = 88

previews = RenderCache()

router = APIRouter(prefix="/api/documents", tags=["documents"])


@lru_cache
def _store_for(data_dir: Path) -> DocumentStore:
    return DocumentStore(data_dir)


def get_store(settings: Annotated[Settings, Depends(get_settings)]) -> DocumentStore:
    return _store_for(settings.data_dir)


Store = Annotated[DocumentStore, Depends(get_store)]


def load(store: DocumentStore, doc_id: str) -> Document:
    try:
        return store.get(doc_id)
    except DocumentNotFoundError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such document.") from None


@router.post(
    "",
    operation_id="createDocument",
    status_code=status.HTTP_201_CREATED,
    responses={413: {"description": "File too large"}, 415: {"description": "Not a photo"}},
)
async def create_document(file: UploadFile, store: Store) -> DocumentView:
    """Upload a JPEG, PNG, or HEIC photo to start editing it."""
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, "Photos up to 64 MB are supported.")
    try:
        doc = store.create(file.filename or "photo", data)
    except imaging.UnsupportedImageError as exc:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, str(exc)) from None
    return DocumentView.of(doc)


@router.get("/{doc_id}", operation_id="getDocument")
def get_document(doc_id: str, store: Store) -> DocumentView:
    return DocumentView.of(load(store, doc_id))


@router.post("/{doc_id}/undo", operation_id="undo")
def undo(doc_id: str, store: Store) -> DocumentView:
    """Step back one agent turn."""
    doc = load(store, doc_id)
    if doc.undo():
        store.save(doc)
    return DocumentView.of(doc)


@router.post("/{doc_id}/redo", operation_id="redo")
def redo(doc_id: str, store: Store) -> DocumentView:
    """Re-apply the most recently undone agent turn."""
    doc = load(store, doc_id)
    if doc.redo():
        store.save(doc)
    return DocumentView.of(doc)


@router.get(
    "/{doc_id}/original",
    operation_id="getOriginal",
    response_class=Response,
    responses={200: {"content": {"image/jpeg": {}}}},
)
def original(doc_id: str, store: Store) -> Response:
    """The unedited photo at preview size, as JPEG (browsers cannot show HEIC)."""
    load(store, doc_id)
    pixels = store.image(doc_id).proxy
    return Response(
        imaging.encode_jpeg(pixels, PREVIEW_QUALITY),
        media_type="image/jpeg",
        headers={"Cache-Control": "private, max-age=31536000, immutable"},
    )


@router.get(
    "/{doc_id}/preview",
    operation_id="getPreview",
    response_class=Response,
    responses={200: {"content": {"image/jpeg": {}}}},
)
def preview(doc_id: str, store: Store) -> Response:
    """The photo with the current edits applied, at preview size, as JPEG.

    Add `?revision=<document revision>` to make the URL unique per edit state; the response
    is the current state either way.
    """
    doc = load(store, doc_id)
    return Response(
        imaging.encode_jpeg(render_preview(store, doc), PREVIEW_QUALITY),
        media_type="image/jpeg",
        headers={"Cache-Control": "no-cache"},
    )


@router.post(
    "/{doc_id}/export",
    operation_id="exportDocument",
    response_class=Response,
    responses={200: {"content": {"image/jpeg": {}, "image/png": {}}}},
)
async def export(doc_id: str, options: ExportOptions, store: Store) -> Response:
    """Render the edits at full resolution and return the file to download."""
    doc = load(store, doc_id)
    loaded = store.image(doc_id)
    data = await asyncio.to_thread(export_bytes, doc, loaded, options)
    filename = export_filename(doc, options)
    return Response(
        data,
        media_type="image/png" if options.format == "png" else "image/jpeg",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quote(filename)}"},
    )


def render_preview(store: DocumentStore, doc: Document) -> imaging.Array:
    loaded = store.image(doc.id)
    return previews.get_or_render(doc.id, loaded.proxy, doc.operations, loaded.proxy_context)


def get_model(settings: Annotated[Settings, Depends(get_settings)]) -> ModelClient | None:
    key = settings.anthropic_api_key
    return ClaudeModel(key.get_secret_value(), settings.anthropic_model) if key else None


@router.websocket("/{doc_id}/chat")
async def chat(
    ws: WebSocket,
    doc_id: str,
    store: Store,
    model: Annotated[ModelClient | None, Depends(get_model)],
) -> None:
    """Chat with the agent about one document.

    The client sends `{"type": "message", "text": "..."}`. For each message the server streams
    `turn_started`, then `text` deltas and `operation` notices as the agent works, then `done`
    with the updated document (or `error`).
    """
    await ws.accept()
    try:
        store.get(doc_id)
    except DocumentNotFoundError:
        await ws.close(code=4404, reason="No such document.")
        return

    async def emit(event: AgentEvent) -> None:
        await ws.send_text(event.model_dump_json())

    agent = AgentService(store, previews, model) if model else None
    try:
        while True:
            incoming = await ws.receive_json()
            text = str(incoming.get("text", "")).strip() if isinstance(incoming, dict) else ""
            if not text:
                await emit(AgentError(message="Type what you would like to change."))
                continue
            if agent is None:
                await emit(
                    AgentError(
                        message="The agent needs a Claude API key. Set ANTHROPIC_API_KEY in "
                        ".env at the repo root and restart the server."
                    )
                )
                continue
            try:
                await agent.run_turn(doc_id, text, emit)
            except WebSocketDisconnect:
                raise
            except Exception as exc:  # Report, keep the socket open for the next message.
                await emit(AgentError(message=f"Something went wrong: {exc}"))
    except WebSocketDisconnect:
        pass
