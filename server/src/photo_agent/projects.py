"""Project files: the original photo plus its edit graph in one file, to save and reopen.

A project file is a zip archive with the extension `.photoagent`:

    manifest.json    {"format": "photo-agent-project", "version": 1}
    document.json    the edit graph (history, layers, chat), as stored on the server
    original.<ext>   the untouched original photo

A single file is easy to keep, move, and back up, and the original inside is never
re-encoded, so reopening a project is lossless.
"""

from __future__ import annotations

import io
import json
import zipfile

from pydantic import ValidationError

from photo_agent.graph import Document

EXTENSION = ".photoagent"
MEDIA_TYPE = "application/vnd.photo-agent.project+zip"
FORMAT = "photo-agent-project"
VERSION = 1
MAX_ORIGINAL_BYTES = 64 * 1024 * 1024
MAX_DOCUMENT_BYTES = 16 * 1024 * 1024
MAX_FILE_BYTES = MAX_ORIGINAL_BYTES + MAX_DOCUMENT_BYTES + 64 * 1024


class ProjectError(ValueError):
    """The file is not a project this version can open."""


def pack(doc: Document, original: bytes, extension: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        manifest = {"format": FORMAT, "version": VERSION}
        zf.writestr("manifest.json", json.dumps(manifest), zipfile.ZIP_DEFLATED)
        zf.writestr("document.json", doc.model_dump_json(indent=2), zipfile.ZIP_DEFLATED)
        # Photos are already compressed; storing them as-is is fast and loses nothing.
        zf.writestr(f"original{extension}", original, zipfile.ZIP_STORED)
    return buf.getvalue()


def unpack(data: bytes) -> tuple[bytes, Document]:
    """The original photo bytes and the document from a project file."""
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        raise ProjectError("This is not a photo-agent project file.") from None
    with zf:
        names = {info.filename: info for info in zf.infolist()}
        manifest = _read(zf, names, "manifest.json", 64 * 1024)
        try:
            meta = json.loads(manifest)
        except ValueError:
            meta = None
        if not isinstance(meta, dict) or meta.get("format") != FORMAT:
            raise ProjectError("This is not a photo-agent project file.")
        if meta.get("version", 0) > VERSION:
            raise ProjectError("This project was saved by a newer version of photo-agent.")
        originals = [n for n in names if n.startswith("original.")]
        if len(originals) != 1:
            raise ProjectError("The project file has no photo in it.")
        original = _read(zf, names, originals[0], MAX_ORIGINAL_BYTES)
        doc = parse_document(_read(zf, names, "document.json", MAX_DOCUMENT_BYTES))
    return original, doc


def parse_document(raw: bytes | str) -> Document:
    try:
        return Document.model_validate_json(raw)
    except ValidationError:
        raise ProjectError("The project's edits could not be read.") from None


def _read(zf: zipfile.ZipFile, names: dict[str, zipfile.ZipInfo], name: str, limit: int) -> bytes:
    info = names.get(name)
    if info is None:
        raise ProjectError(f"The project file is missing {name}.")
    if info.file_size > limit:
        raise ProjectError("The project file is too large.")
    with zf.open(info) as f:
        data = f.read(limit + 1)
    if len(data) > limit:
        raise ProjectError("The project file is too large.")
    return data
