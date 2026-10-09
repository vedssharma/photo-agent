"""Semantic masks for one document: find what a mask selects, once, and remember it.

A semantic mask says what to select ("the sky", the object in a box); the model worker finds
it in the framed original photo. Results are cached by everything they depend on (what is
selected, the framing, and the backend that finds it), in memory and as PNGs in the
document's folder, so re-rendering, undo, and export never run a model twice. Change the
crop or the selection and the key changes, so the model runs again on the new framing.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
from PIL import Image

from photo_agent import imaging
from photo_agent.imaging import Array
from photo_agent.masks import SemanticMask
from photo_agent.operations import OpBase
from photo_agent.vision.backends import BackendMode
from photo_agent.vision.tasks import TASKS
from photo_agent.vision.worker import JobFailedError, ModelWorker

log = logging.getLogger(__name__)

WORK_EDGE = 1024
"""Long edge of the framed photo that models look at."""
MEMORY_ITEMS = 32

TARGET_TASKS = {
    "subject": "segment_subject",
    "people": "segment_people",
    "sky": "segment_sky",
    "object": "segment_object",
}
FACE_TASK = "parse_face"
"""Every face part (skin, eyes, ...) comes from one face-parsing job."""


def task_for(mask: SemanticMask) -> str:
    return TARGET_TASKS.get(mask.target, FACE_TASK)


class DocumentVision:
    """Runs and caches the models behind one document's semantic masks. Thread-safe."""

    def __init__(
        self,
        doc_id: str,
        proxy: Array,
        source_aspect: float,
        scale: float,
        worker: ModelWorker,
        folder: Path,
        backends: BackendMode = "auto",
    ) -> None:
        self.doc_id = doc_id
        self.proxy = proxy
        self.source_aspect = source_aspect
        self.scale = scale
        self.worker = worker
        self.folder = folder
        self.backends = backends
        self._memory: OrderedDict[str, Array] = OrderedDict()
        self._locks: dict[str, threading.Lock] = {}
        self._lock = threading.Lock()

    def mask(self, mask: SemanticMask, framing: Sequence[OpBase], shape: tuple[int, int]) -> Array:
        raster = self.selection(mask, framing)
        h, w = shape
        if raster.shape != (h, w):
            raster = np.asarray(
                cv2.resize(raster, (w, h), interpolation=cv2.INTER_LINEAR), np.float32
            )
        return raster

    def selection(self, mask: SemanticMask, framing: Sequence[OpBase]) -> Array:
        """What the mask selects, at the models' working size of the framed photo."""
        task = task_for(mask)
        key = self.key(mask, framing)
        name = key if task != FACE_TASK else f"{key}-{mask.target}"
        found = self._remembered(name)
        if found is not None:
            return found
        with self._key_lock(key):
            found = self._remembered(name)
            if found is not None:
                return found
            image = self.framed(framing)
            params: dict[str, Any] = {}
            if task == "segment_object":
                params = {
                    "box": mask.box,
                    "points": [[p.x, p.y, p.include] for p in mask.points],
                }
            label = f"Selecting {mask.description}" if mask.description else None
            try:
                result = self.worker.run(task, image, params, doc_id=self.doc_id, label=label)
            except JobFailedError as exc:
                # Leave the layer without effect rather than failing the whole render.
                log.warning("Could not find %s: %s", mask.target, exc)
                return np.zeros(image.shape[:2], np.float32)
            if task == FACE_TASK:
                for part, raster in result.value.items():
                    self._store(f"{key}-{part}", raster)
            else:
                self._store(name, result.value)
        stored = self._remembered(name)
        return stored if stored is not None else np.zeros(image.shape[:2], np.float32)

    def key(self, mask: SemanticMask, framing: Sequence[OpBase]) -> str:
        task = task_for(mask)
        choices = TASKS[task].choices(self.backends)
        payload: dict[str, Any] = {
            "v": 1,
            "task": task,
            "backend": choices[0].name if choices else "none",
            "framing": [op.model_dump(exclude={"id"}) for op in framing],
        }
        if task == "segment_object":
            payload["box"] = mask.box
            payload["points"] = [p.model_dump() for p in mask.points]
        elif task != FACE_TASK:
            payload["target"] = mask.target
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        return digest[:24]

    def framed(self, framing: Sequence[OpBase]) -> Array:
        """The original with the framing applied, at the models' working size."""
        from photo_agent.render import RenderContext, apply_operations

        ctx = RenderContext(scale=self.scale, source_aspect=self.source_aspect)
        out = apply_operations(self.proxy.astype(np.float32, copy=True), framing, ctx)
        return np.ascontiguousarray(
            imaging.resize_long_edge(np.clip(out, 0.0, 1.0), WORK_EDGE), np.float32
        )

    # Cache

    def _key_lock(self, key: str) -> threading.Lock:
        with self._lock:
            return self._locks.setdefault(key, threading.Lock())

    def _path(self, name: str) -> Path:
        return self.folder / f"{name}.png"

    def _remembered(self, name: str) -> Array | None:
        with self._lock:
            hit = self._memory.get(name)
            if hit is not None:
                self._memory.move_to_end(name)
                return hit
        path = self._path(name)
        if not path.is_file():
            return None
        try:
            with Image.open(path) as img:
                raster = np.asarray(img.convert("L"), np.float32) / 255.0
        except OSError:
            return None
        self._remember(name, raster)
        return raster

    def _store(self, name: str, raster: Any) -> None:
        values = np.clip(np.asarray(raster, np.float32), 0.0, 1.0)
        self.folder.mkdir(parents=True, exist_ok=True)
        tmp = self._path(name).with_suffix(".tmp")
        tmp.write_bytes(imaging.encode_gray_png(values))
        tmp.replace(self._path(name))
        # Keep what was saved (8-bit), so a fresh process renders exactly the same.
        self._remember(name, imaging.to_uint8(values).astype(np.float32) / 255.0)

    def _remember(self, name: str, raster: Array) -> None:
        with self._lock:
            self._memory[name] = cast(Array, raster.astype(np.float32))
            self._memory.move_to_end(name)
            while len(self._memory) > MEMORY_ITEMS:
                self._memory.popitem(last=False)
