"""HTTP routes for the photo library: import, browse, and the indexing that runs behind it."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, UploadFile, status
from fastapi.responses import Response
from pydantic import BaseModel, Field

from photo_agent import imaging
from photo_agent.library import (
    ImportResult,
    LibraryPhoto,
    PhotoNotFoundError,
    Skipped,
    culling,
    duplicates,
    faces,
    organize,
    tagging,
)
from photo_agent.library import search as searching
from photo_agent.library.facets import Facets, facets
from photo_agent.library.filters import PhotoFilter
from photo_agent.library.indexer import IndexStatus
from photo_agent.routes import MAX_UPLOAD_BYTES, AdvisorDep, LibraryDep

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
def list_photos(
    library: LibraryDep, filters: Annotated[PhotoFilter, Query()]
) -> list[LibraryPhoto]:
    """The library's photos, newest import first, narrowed down by any filters given."""
    return library.photos(filters)


@router.get("/facets", operation_id="getFacets")
def get_facets(library: LibraryDep) -> Facets:
    """The cameras, places, and years in the library, with how many photos each has."""
    return facets(library.db)


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


class Marked(BaseModel):
    changed: int
    photos: list[LibraryPhoto]


@router.post("/photos/marks", operation_id="markPhotos")
def mark_photos(body: organize.Marks, library: LibraryDep) -> Marked:
    """Rate, flag (pick or reject), or color-label several photos at once."""
    changed = organize.mark(library.db, body)
    wanted = set(body.ids)
    return Marked(changed=changed, photos=[p for p in library.photos() if p.id in wanted])


class TagCount(BaseModel):
    tag: str
    count: int


@router.get("/tags", operation_id="listTags")
def list_tags(library: LibraryDep) -> list[TagCount]:
    """Every tag in use, found or added by hand, most used first."""
    return [TagCount(tag=tag, count=n) for tag, n in tagging.all_tags(library.db)]


@router.post("/photos/tags", operation_id="tagPhotos")
def tag_photos(body: tagging.TagChange, library: LibraryDep) -> list[LibraryPhoto]:
    """Add or remove tags by hand. A found tag that is removed stays removed."""
    tagging.change_tags(library.db, body)
    wanted = set(body.ids)
    return [p for p in library.photos() if p.id in wanted]


@router.post("/search", operation_id="searchLibrary")
async def search_library(
    body: searching.SearchRequest, library: LibraryDep, advisor: AdvisorDep
) -> searching.SearchResult:
    """Search in everyday words ("beach photos from last summer with the dog"). The query
    is read into filters (by Claude when an API key is set, else by built-in rules) and the
    answer says how it was understood."""
    return await searching.search(library, body, advisor)


def _album[R](fn: Callable[[], R]) -> R:
    try:
        return fn()
    except organize.AlbumNotFoundError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such album.") from None


@router.get("/albums", operation_id="listAlbums")
def list_albums(library: LibraryDep) -> list[organize.Album]:
    """Every album, by name."""
    return organize.albums(library.db)


@router.post("/albums", operation_id="createAlbum", status_code=status.HTTP_201_CREATED)
def create_album(body: organize.NewAlbum, library: LibraryDep) -> organize.Album:
    """Make an album, optionally with photos in it already."""
    return organize.create_album(library.db, body)


@router.patch(
    "/albums/{album_id}",
    operation_id="renameAlbum",
    responses={404: {"description": "No such album"}},
)
def rename_album(album_id: str, body: organize.AlbumChange, library: LibraryDep) -> organize.Album:
    return _album(lambda: organize.rename_album(library.db, album_id, body))


@router.delete(
    "/albums/{album_id}",
    operation_id="deleteAlbum",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={404: {"description": "No such album"}},
)
def delete_album(album_id: str, library: LibraryDep) -> None:
    """Delete an album. Its photos stay in the library."""
    _album(lambda: organize.delete_album(library.db, album_id))


@router.post(
    "/albums/{album_id}/photos",
    operation_id="addToAlbum",
    responses={404: {"description": "No such album"}},
)
def add_to_album(album_id: str, body: PhotoIds, library: LibraryDep) -> organize.Album:
    _album(lambda: organize.add_to_album(library.db, album_id, body.ids))
    return organize.album(library.db, album_id)


@router.post(
    "/albums/{album_id}/photos/remove",
    operation_id="removeFromAlbum",
    responses={404: {"description": "No such album"}},
)
def remove_from_album(album_id: str, body: PhotoIds, library: LibraryDep) -> organize.Album:
    """Take photos out of an album. They stay in the library."""
    _album(lambda: organize.remove_from_album(library.db, album_id, body.ids))
    return organize.album(library.db, album_id)


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


@router.post("/cull", operation_id="cullPhotos")
async def cull_photos(
    body: culling.CullRequest, library: LibraryDep, advisor: AdvisorDep
) -> culling.CullResult:
    """Pick the best photos from a shoot: one per moment (burst or series of the same
    thing), judged on sharpness, exposure, eyes open, and appeal, with the reasons for
    each. Claude writes the explanation when an API key is set."""
    return await culling.cull(library, body, advisor)


@router.get("/duplicates", operation_id="listDuplicates")
def list_duplicates(library: LibraryDep) -> list[duplicates.DuplicateGroup]:
    """Photos that are the same photo, or the same shot, each group with one suggested to
    keep. Photos still being indexed are not compared yet."""
    return duplicates.groups(library)


@router.post(
    "/duplicates/dismiss",
    operation_id="dismissDuplicates",
    status_code=status.HTTP_204_NO_CONTENT,
)
def dismiss_duplicates(body: duplicates.NotDuplicates, library: LibraryDep) -> None:
    """These photos are different, however alike: stop grouping them as duplicates."""
    duplicates.mark_not_duplicates(library.db, body.ids)


class LibrarySettings(BaseModel):
    faces: bool = Field(
        description="People albums: find and group faces, on this computer only. Turning it "
        "off forgets every face and person."
    )


@router.get("/settings", operation_id="getLibrarySettings")
def get_settings(library: LibraryDep) -> LibrarySettings:
    return LibrarySettings(faces=faces.enabled(library.db))


@router.put("/settings", operation_id="changeLibrarySettings")
async def change_settings(body: LibrarySettings, library: LibraryDep) -> LibrarySettings:
    faces.set_enabled(library.db, body.faces)
    # Either way every photo's index changes: faces to find, or just a new index key.
    await asyncio.to_thread(library.indexer.kick)
    return LibrarySettings(faces=faces.enabled(library.db))


def _people[R](fn: Callable[[], R]) -> R:
    try:
        return fn()
    except faces.PersonNotFoundError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such person.") from None
    except faces.FaceNotFoundError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such face.") from None


@router.get("/people", operation_id="listPeople")
def list_people(library: LibraryDep) -> list[faces.Person]:
    """Everyone found in the library's photos: named people first, then by photo count."""
    return faces.people(library.db)


@router.patch(
    "/people/{person_id}",
    operation_id="renamePerson",
    responses={404: {"description": "No such person"}},
)
def rename_person(person_id: str, body: faces.PersonChange, library: LibraryDep) -> faces.Person:
    return _people(lambda: faces.rename(library.db, person_id, body))


class Merge(BaseModel):
    into: str = Field(description="The person to keep.")


@router.post(
    "/people/{person_id}/merge",
    operation_id="mergePeople",
    responses={404: {"description": "No such person"}},
)
def merge_people(person_id: str, body: Merge, library: LibraryDep) -> faces.Person:
    """The same person found twice: their faces all go to `into`."""
    return _people(lambda: faces.merge(library.db, person_id, body.into))


@router.delete(
    "/people/{person_id}",
    operation_id="forgetPerson",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={404: {"description": "No such person"}},
)
def forget_person(person_id: str, library: LibraryDep) -> None:
    """Stop showing this person. Their faces are not grouped again."""
    _people(lambda: faces.forget(library.db, person_id))


@router.get(
    "/photos/{photo_id}/faces",
    operation_id="listFaces",
    responses={404: {"description": "No such photo"}},
)
def list_faces(photo_id: str, library: LibraryDep) -> list[faces.Face]:
    """The faces found in one photo and who each one is."""
    _photo(library, photo_id)
    return faces.faces_in(library.db, photo_id)


@router.post(
    "/faces/{face_id}",
    operation_id="changeFace",
    responses={404: {"description": "No such face or person"}},
)
def change_face(face_id: str, body: faces.FaceChange, library: LibraryDep) -> faces.Face:
    """Say who a face is, or that it is nobody in particular ("not this person")."""
    return _people(lambda: faces.move_face(library.db, face_id, body))


@router.get(
    "/faces/{face_id}/thumbnail",
    operation_id="getFaceThumbnail",
    response_class=Response,
    responses={200: {"content": {"image/jpeg": {}}}, 404: {"description": "No such face"}},
)
async def face_thumbnail(face_id: str, library: LibraryDep) -> Response:
    data = await asyncio.to_thread(_people, lambda: faces.face_thumbnail(library, face_id))
    return Response(
        data, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=86400"}
    )


@router.get("/status", operation_id="getLibraryStatus")
def library_status(library: LibraryDep) -> IndexStatus:
    """How far indexing has got: the web app polls this while photos are being indexed."""
    return library.indexer.status()
