"""HTTP routes for the photo library: import, browse, and the indexing that runs behind it."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, UploadFile, status
from fastapi.responses import Response
from pydantic import BaseModel, Field

from photo_agent import imaging
from photo_agent.library import ImportResult, LibraryPhoto, PhotoNotFoundError, Skipped
from photo_agent.library.indexer import IndexStatus
from photo_agent.routes import MAX_UPLOAD_BYTES, LibraryDep

router = APIRouter(prefix="/api/library", tags=["library"])

MAX_FILES_PER_IMPORT = 100
"""Files per import request; the web app sends a big import in several requests."""


def _photo(library: LibraryDep, photo_id: str) -> LibraryPhoto:
    try:
        return library.get(photo_id)
    except PhotoNotFoundError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such photo in the library.") from None


@router.post(
    "/photos",
    operation_id="importPhotos",
    status_code=status.HTTP_201_CREATED,
    responses={413: {"description": "Too many files at once"}},
)
async def import_photos(files: list[UploadFile], library: LibraryDep) -> ImportResult:
    """Add photos to the library, many at once (a whole folder is sent a batch at a time).
    Files that are not photos, are too large, or are already in the library are skipped
    and listed with the reason. New photos are indexed in the background."""
    if len(files) > MAX_FILES_PER_IMPORT:
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE,
            f"Send at most {MAX_FILES_PER_IMPORT} files per request.",
        )
    result = ImportResult(imported=[], skipped=[])
    for file in files:
        name = file.filename or "photo"
        data = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            result.skipped.append(Skipped(filename=name, reason="Larger than 64 MB."))
            continue
        try:
            outcome = await asyncio.to_thread(library.import_file, name, data)
        except imaging.UnsupportedImageError as exc:
            result.skipped.append(Skipped(filename=name, reason=str(exc)))
            continue
        if isinstance(outcome, LibraryPhoto):
            result.imported.append(outcome)
        else:
            result.skipped.append(outcome)
    if result.imported:
        await asyncio.to_thread(library.indexer.kick)
    return result


@router.get("/photos", operation_id="listLibraryPhotos")
def list_photos(library: LibraryDep) -> list[LibraryPhoto]:
    """Every photo in the library, newest import first."""
    return library.photos()


@router.get(
    "/photos/{photo_id}",
    operation_id="getLibraryPhoto",
    responses={404: {"description": "No such photo"}},
)
def get_photo(photo_id: str, library: LibraryDep) -> LibraryPhoto:
    return _photo(library, photo_id)


class PhotoIds(BaseModel):
    ids: list[str] = Field(min_length=1, max_length=5000)


class Removed(BaseModel):
    removed: int


@router.post("/photos/remove", operation_id="removeLibraryPhotos")
def remove_photos(body: PhotoIds, library: LibraryDep) -> Removed:
    """Delete photos from the library, with their edits. This cannot be undone."""
    return Removed(removed=library.remove(body.ids))


@router.get(
    "/photos/{photo_id}/thumbnail",
    operation_id="getThumbnail",
    response_class=Response,
    responses={200: {"content": {"image/jpeg": {}}}, 404: {"description": "No such photo"}},
)
async def thumbnail(photo_id: str, library: LibraryDep) -> Response:
    """A small JPEG of the photo with its current edits. Add `?revision=` (the photo's
    revision) so the browser can keep it until the photo is edited again."""
    _photo(library, photo_id)
    data = await asyncio.to_thread(library.thumbnail, photo_id)
    return Response(
        data, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=86400"}
    )


@router.get("/status", operation_id="getLibraryStatus")
def library_status(library: LibraryDep) -> IndexStatus:
    """How far indexing has got: the web app polls this while photos are being indexed."""
    return library.indexer.status()
