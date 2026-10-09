"""The tasks the model worker can run, each with its backends in order of preference."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from photo_agent.vision import classical, hub
from photo_agent.vision.backends import Backend, Env, Progress, SelfTest, Task


class Classical:
    """A classical computer-vision fallback (see `classical.py`)."""

    name = "classical"
    license = "n/a (no model weights)"
    uses_weights = False

    def __init__(self, fn: Callable[..., Any]) -> None:
        self.fn = fn

    def available(self) -> bool:
        return True

    def run(self, image: Any, params: dict[str, Any], progress: Progress, env: Env) -> Any:
        progress(None, "Working")
        return self.fn(image, **params)


def _tasks() -> list[Task]:
    def task(name: str, label: str, *backends: Backend) -> Task:
        return Task(name, label, list(backends))

    return [
        task("self_test", "Checking the model worker", SelfTest()),
        task(
            "segment_subject",
            "Finding the main subject",
            hub.BiRefNetSubject(),
            Classical(classical.subject),
        ),
        task(
            "segment_people",
            "Finding people",
            hub.AdeSegmenter("people"),
            Classical(classical.people),
        ),
        task("segment_sky", "Finding the sky", hub.AdeSegmenter("sky"), Classical(classical.sky)),
        task(
            "segment_object",
            "Selecting",
            hub.Sam2Object(),
            Classical(classical.object_at),
        ),
        task("parse_face", "Finding faces", hub.FaceParser(), Classical(classical.face_parts)),
        task("inpaint", "Filling in", hub.LamaInpaint(), Classical(classical.inpaint)),
        task("generate", "Generating", hub.SdxlInpaint(), Classical(classical.generate)),
        task("relight", "Relighting", hub.IcLight(), Classical(classical.relight)),
        task("upscale", "Enlarging", hub.RealEsrgan(), Classical(classical.upscale)),
        task("restore_faces", "Restoring faces", hub.Gfpgan(), Classical(classical.restore_faces)),
        task("colorize", "Colorizing", hub.DdColor(), Classical(classical.colorize)),
    ]


TASKS: dict[str, Task] = {task.name: task for task in _tasks()}
