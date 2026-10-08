"""HTTP routes for documents: upload, inspect, undo/redo, and preview images."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from fastapi.responses import Response

from photo_agent import imaging
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


def render_preview(store: DocumentStore, doc: Document) -> imaging.Array:
    loaded = store.image(doc.id)
    return previews.get_or_render(doc.id, loaded.proxy, doc.operations, loaded.proxy_context)
