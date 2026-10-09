"""Backends: what actually runs a task, model or classical.

Each task lists its backends in order of preference. The first one that is available wins:
model backends need their optional packages (`uv sync --extra models`) and can download their
weights; the last backend of every task is classical computer vision with no weights, so a
task always has something to run. A model backend that fails (no network for the download,
out of memory) is set aside for the rest of the process and the next one is used.

The backend's name is part of every cached result's key, so switching backends never mixes
results from different models.
"""

from __future__ import annotations

import importlib.util
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

log = logging.getLogger(__name__)

Progress = Callable[[float | None, str], None]
"""Report progress: a fraction from 0 to 1 (or None if unknown) and what is happening."""

BackendMode = Literal["auto", "classical"]
Device = Literal["auto", "cpu", "cuda", "mps"]


@dataclass(frozen=True)
class WorkerConfig:
    """What a worker needs to pick and load backends. Sent to the worker process."""

    backends: BackendMode = "auto"
    device: Device = "auto"
    cache_dir: Path = Path(".data/models")


@dataclass
class Env:
    """Where a backend runs, resolved inside the worker."""

    device: str
    cache_dir: Path


class Backend(Protocol):
    name: str
    """Identifies the model (or "classical"); recorded with and keys every result."""
    license: str
    """The model weights' license, tracked so a later product does not need a rewrite."""
    uses_weights: bool

    def available(self) -> bool:
        """Whether this backend's packages are installed."""
        ...

    def run(self, image: Any, params: dict[str, Any], progress: Progress, env: Env) -> Any:
        """Do the work. `image` is RGB float32 in 0..1 unless the task says otherwise."""
        ...


def has_packages(*names: str) -> bool:
    return all(importlib.util.find_spec(name) is not None for name in names)


@dataclass
class Task:
    name: str
    label: str
    """What the person sees while it runs, e.g. "Finding the sky"."""
    backends: list[Backend]
    broken: set[str] = field(default_factory=set)
    """Backends that failed in this process, skipped from then on."""

    def choices(self, mode: BackendMode) -> list[Backend]:
        """Backends to try, best first."""
        usable = [
            b
            for b in self.backends
            if b.name not in self.broken and (mode == "auto" or not b.uses_weights)
        ]
        return [b for b in usable if b.available()]


def pick_device(preference: Device) -> str:
    if preference != "auto":
        return preference
    if not has_packages("torch"):
        return "cpu"
    import torch

    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


class SelfTest:
    """Checks that the worker runs jobs and reports progress: returns the image's mean."""

    name = "self-test"
    license = "n/a"
    uses_weights = False

    def available(self) -> bool:
        return True

    def run(self, image: Any, params: dict[str, Any], progress: Progress, env: Env) -> Any:
        import numpy as np

        progress(0.5, "Checking")
        if params.get("fail"):
            raise RuntimeError("Asked to fail.")
        return float(np.asarray(image).mean())
