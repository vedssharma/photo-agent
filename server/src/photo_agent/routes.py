"""HTTP routes for documents: upload, inspect, history, previews, chat, and export."""

from __future__ import annotations

import asyncio
from functools import lru_cache
from pathlib import Path
from typing import Annotated
from urllib.parse import quote

from fastapi import (
    APIRouter,
    Body,
    Depends,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field

from photo_agent import generative, geometry, imaging, portrait, projects, recipes
from photo_agent.agent import AgentError, AgentEvent, AgentService, ClaudeModel, ModelClient
from photo_agent.export import ExportOptions, export_bytes, export_filename
from photo_agent.graph import Document, DocumentView
from photo_agent.layers import EditState, Layer
from photo_agent.render import RenderCache, render, render_layer_mask
from photo_agent.settings import Settings, get_settings
from photo_agent.store import DocumentNotFoundError, DocumentStore, MismatchError
from photo_agent.vision.backends import BackendMode, WorkerConfig
from photo_agent.vision.worker import JobStatus, Mode, ModelWorker

MAX_UPLOAD_BYTES = 64 * 1024 * 1024
PREVIEW_QUALITY = 88

previews = RenderCache()

router = APIRouter(prefix="/api/documents", tags=["documents"])
projects_router = APIRouter(prefix="/api/projects", tags=["projects"])
recipes_router = APIRouter(prefix="/api/recipes", tags=["recipes"])


@lru_cache
def _worker_for(config: WorkerConfig, mode: Mode) -> ModelWorker:
    return ModelWorker(config, mode)


def get_worker(settings: Annotated[Settings, Depends(get_settings)]) -> ModelWorker:
    config = WorkerConfig(
        backends=settings.model_backends,
        device=settings.model_device,
        cache_dir=settings.model_cache_dir,
    )
    return _worker_for(config, settings.model_worker)


Worker = Annotated[ModelWorker, Depends(get_worker)]


@lru_cache
def _store_for(data_dir: Path, worker: ModelWorker, backends: BackendMode) -> DocumentStore:
    return DocumentStore(data_dir, worker=worker, backends=backends)


def get_store(settings: Annotated[Settings, Depends(get_settings)]) -> DocumentStore:
    return _store_for(settings.data_dir, get_worker(settings), settings.model_backends)


Store = Annotated[DocumentStore, Depends(get_store)]


@lru_cache
def _recipes_for(data_dir: Path) -> recipes.RecipeStore:
    return recipes.RecipeStore(data_dir)


def get_recipes(settings: Annotated[Settings, Depends(get_settings)]) -> recipes.RecipeStore:
    return _recipes_for(settings.data_dir)


Recipes = Annotated[recipes.RecipeStore, Depends(get_recipes)]


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


@projects_router.post(
    "",
    operation_id="openProject",
    status_code=status.HTTP_201_CREATED,
    responses={
        413: {"description": "File too large"},
        415: {"description": "Not a project or photo"},
    },
)
async def open_project(
    store: Store,
    file: UploadFile | None = None,
    original: UploadFile | None = None,
    graph: Annotated[str | None, Form()] = None,
) -> DocumentView:
    """Reopen a saved project: either a `.photoagent` project `file`, or an `original`
    photo plus its `graph` (the document JSON), as the browser keeps them for autosave.

    The project keeps its id unless another document already has it."""
    try:
        if file is not None:
            data = await _read_limited(
                file, projects.MAX_ORIGINAL_BYTES + projects.MAX_DOCUMENT_BYTES
            )
            source, edits = projects.unpack(data)
        elif original is not None and graph is not None:
            source = await _read_limited(original, MAX_UPLOAD_BYTES)
            edits = projects.parse_document(graph)
        else:
            raise projects.ProjectError("Send a project file, or a photo and its edits.")
        doc = store.create(edits.filename, source, edits)
    except (projects.ProjectError, MismatchError, imaging.UnsupportedImageError) as exc:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, str(exc)) from None
    return DocumentView.of(doc)


async def _read_limited(file: UploadFile, limit: int) -> bytes:
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, "That file is too large.")
    return data


@router.get("/{doc_id}", operation_id="getDocument")
def get_document(doc_id: str, store: Store) -> DocumentView:
    return DocumentView.of(load(store, doc_id))


@router.get(
    "/{doc_id}/project",
    operation_id="downloadProject",
    response_class=Response,
    responses={200: {"content": {projects.MEDIA_TYPE: {}}}},
)
def download_project(doc_id: str, store: Store) -> Response:
    """The original photo and all its edits as one `.photoagent` file, to keep and reopen."""
    doc = load(store, doc_id)
    original = store.original_file(doc_id)
    data = projects.pack(doc, original.read_bytes(), original.suffix)
    stem = doc.filename.rsplit(".", 1)[0] or "photo"
    filename = f"{stem}{projects.EXTENSION}"
    return Response(
        data,
        media_type=projects.MEDIA_TYPE,
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quote(filename)}"},
    )


@router.get("/{doc_id}/graph", operation_id="getGraph")
def get_graph(doc_id: str, store: Store) -> Document:
    """The full stored document (history, layers, chat), as kept for browser autosave."""
    return load(store, doc_id)


@router.get(
    "/{doc_id}/source",
    operation_id="getSource",
    response_class=Response,
    responses={200: {"content": {"image/jpeg": {}, "image/png": {}, "image/heic": {}}}},
)
def get_source(doc_id: str, store: Store) -> Response:
    """The original file exactly as uploaded."""
    load(store, doc_id)
    path = store.original_file(doc_id)
    media = {".jpg": "image/jpeg", ".png": "image/png", ".heic": "image/heic"}
    return Response(
        path.read_bytes(),
        media_type=media.get(path.suffix, "application/octet-stream"),
        headers={"Cache-Control": "private, max-age=31536000, immutable"},
    )


@router.post("/{doc_id}/undo", operation_id="undo")
def undo(doc_id: str, store: Store) -> DocumentView:
    """Step back one step in the history."""
    doc = load(store, doc_id)
    if doc.undo():
        store.save(doc)
    return DocumentView.of(doc)


@router.post("/{doc_id}/redo", operation_id="redo")
def redo(doc_id: str, store: Store) -> DocumentView:
    """Re-apply the most recently undone step."""
    doc = load(store, doc_id)
    if doc.redo():
        store.save(doc)
    return DocumentView.of(doc)


class ManualEdit(BaseModel):
    label: str = Field(min_length=1, max_length=120, description="Name for the history.")
    state: EditState = Field(description="The complete edit state after the change.")
    coalesce: str | None = Field(
        None,
        max_length=200,
        description="What was tweaked, e.g. one slider. Repeated tweaks with the same key "
        "update the previous step instead of adding another.",
    )


@router.post("/{doc_id}/edits", operation_id="editByHand")
def edit_by_hand(doc_id: str, edit: ManualEdit, store: Store) -> DocumentView:
    """Record a change made with the manual controls as a named step in the history."""
    doc = load(store, doc_id)
    state = generative.stamp(edit.state, store.model_for)
    if doc.edit_by_hand(edit.label, state, edit.coalesce):
        store.save(doc)
        warm_preview(store, doc)
    return DocumentView.of(doc)


@router.post(
    "/{doc_id}/recipes/{recipe_id}",
    operation_id="applyRecipe",
    responses={404: {"description": "No such document or recipe"}},
)
def apply_recipe(doc_id: str, recipe_id: str, store: Store, saved: Recipes) -> DocumentView:
    """Add a recipe's layers on top of the current edits, as one step in the history."""
    doc = load(store, doc_id)
    try:
        recipe = saved.get(recipe_id)
    except recipes.RecipeNotFoundError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such recipe.") from None
    if doc.edit_by_hand(f"Apply recipe “{recipe.name}”", recipes.apply(recipe, doc.state)):
        store.save(doc)
        warm_preview(store, doc)
    return DocumentView.of(doc)


@router.post("/{doc_id}/retouch", operation_id="retouchPortrait")
def retouch_portrait(
    doc_id: str,
    store: Store,
    options: Annotated[portrait.Retouch | None, Body()] = None,
) -> DocumentView:
    """Add portrait retouch layers (skin, eyes, teeth) on top, as one step in the history;
    without options, the subtle defaults."""
    doc = load(store, doc_id)
    state = doc.state
    state.layers.extend(portrait.retouch_layers(options or portrait.Retouch()))
    if doc.edit_by_hand("Retouch portrait", state):
        store.save(doc)
        warm_preview(store, doc)
    return DocumentView.of(doc)


@router.post("/{doc_id}/straighten", operation_id="autoStraighten")
def auto_straighten(
    doc_id: str,
    store: Store,
    options: Annotated[geometry.AutoStraighten | None, Body()] = None,
) -> DocumentView:
    """Level the photo and square up converging verticals, measured from its straight
    lines, as one step in the history. 422 when it finds nothing to go by."""
    doc = load(store, doc_id)
    loaded = store.image(doc_id)
    state = doc.state
    found = geometry.auto_level(
        state.framing,
        lambda framing: render(loaded.proxy, framing, loaded.proxy_context),
        options,
    )
    if not found.describe():
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "No clear horizon or verticals to go by, or the photo is already straight.",
        )
    state.framing = found.framing  # type: ignore[assignment]
    if doc.edit_by_hand("Auto straighten", state):
        store.save(doc)
        warm_preview(store, doc)
    return DocumentView.of(doc)


class Checkout(BaseModel):
    step_id: str | None = Field(description="Step to show, or null for the original photo.")


@router.post(
    "/{doc_id}/checkout",
    operation_id="checkout",
    responses={404: {"description": "No such document or step"}},
)
def checkout(doc_id: str, body: Checkout, store: Store) -> DocumentView:
    """Jump to any step in the history. Editing from there starts a new branch; the steps
    after it stay in the history."""
    doc = load(store, doc_id)
    try:
        changed = doc.checkout(body.step_id)
    except KeyError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such step.") from None
    if changed:
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


@router.get(
    "/{doc_id}/before",
    operation_id="getBefore",
    response_class=Response,
    responses={200: {"content": {"image/jpeg": {}}}},
)
def before(doc_id: str, store: Store) -> Response:
    """The unedited look with the current framing (crop, rotation, flips) applied, so it lines
    up with the preview for before/after comparison."""
    doc = load(store, doc_id)
    loaded = store.image(doc_id)
    framing = EditState(framing=doc.state.framing)
    pixels = previews.get_or_render(doc.id, loaded.proxy, framing, loaded.proxy_context)
    return Response(
        imaging.encode_jpeg(pixels, PREVIEW_QUALITY),
        media_type="image/jpeg",
        headers={"Cache-Control": "no-cache"},
    )


@router.get(
    "/{doc_id}/layers/{layer_id}/mask",
    operation_id="getLayerMask",
    response_class=Response,
    responses={200: {"content": {"image/png": {}}}, 404: {"description": "No such layer"}},
)
def layer_mask(doc_id: str, layer_id: str, store: Store) -> Response:
    """Where a layer of the current state applies, at preview size, as a grayscale PNG
    (white is full effect). Add `?revision=` to make the URL unique per edit state."""
    doc = load(store, doc_id)
    loaded = store.image(doc_id)
    try:
        alpha = render_layer_mask(loaded.proxy, doc.state, layer_id, loaded.proxy_context)
    except KeyError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such layer.") from None
    return Response(
        imaging.encode_gray_png(alpha),
        media_type="image/png",
        headers={"Cache-Control": "no-cache"},
    )


@router.get("/{doc_id}/jobs", operation_id="listJobs")
def list_jobs(doc_id: str, store: Store, worker: Worker) -> list[JobStatus]:
    """AI model jobs running (or just finished) for this document, with their progress.
    The web app polls this while it waits on a change."""
    load(store, doc_id)
    return worker.jobs(doc_id)


def render_preview(store: DocumentStore, doc: Document) -> imaging.Array:
    loaded = store.image(doc.id)
    return previews.get_or_render(doc.id, loaded.proxy, doc.state, loaded.proxy_context)


def warm_preview(store: DocumentStore, doc: Document) -> None:
    """Render the new state before answering, so any model jobs it needs (finding the sky,
    say) run while the change is in flight and show their progress, and the preview the
    browser asks for next is already cached."""
    render_preview(store, doc)


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


@recipes_router.get("", operation_id="listRecipes")
def list_recipes(saved: Recipes) -> list[recipes.Recipe]:
    """Saved recipes, newest first."""
    return saved.all()


class NewRecipe(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    name: str = Field(min_length=1, max_length=80)
    layers: list[Layer] = Field(
        min_length=1,
        description="The layers to keep. Brush masks are dropped, since they only fit the "
        "photo they were painted on.",
    )


@recipes_router.post("", operation_id="createRecipe", status_code=status.HTTP_201_CREATED)
def create_recipe(body: NewRecipe, saved: Recipes) -> recipes.Recipe:
    """Save a set of layers as a recipe to apply to other photos."""
    return saved.create(body.name, body.layers)


@recipes_router.delete(
    "/{recipe_id}",
    operation_id="deleteRecipe",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={404: {"description": "No such recipe"}},
)
def delete_recipe(recipe_id: str, saved: Recipes) -> None:
    try:
        saved.delete(recipe_id)
    except recipes.RecipeNotFoundError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such recipe.") from None
