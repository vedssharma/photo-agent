"""Generative edits: operations whose pixels come from a generative model.

Each one records its prompt, its seed, and the model that generates it, so the same result
renders again (on undo, on export, after reopening a project) and a new seed gives a
different take on the same request. The seed and the model are filled in when the edit is
made (`stamp`), not left to chance at render time.
"""

from __future__ import annotations

import secrets
from collections.abc import Callable
from typing import Any

from photo_agent.layers import EditState
from photo_agent.operations import (
    SEED_MAX,
    Colorize,
    Expand,
    Generate,
    GenerativeBase,
    OpBase,
    Relight,
    ReplaceBackground,
    RestoreFaces,
    Restyle,
)
from photo_agent.vision.worker import PREFER

TASKS: dict[type[OpBase], str] = {
    Generate: "generate",
    Expand: "generate",
    ReplaceBackground: "generate",
    Relight: "relight",
    RestoreFaces: "restore_faces",
    Colorize: "colorize",
    Restyle: "generate",
}
"""The model worker task behind each generative operation."""


def task_for(op: GenerativeBase) -> str:
    return TASKS[type(op)]


def new_seed() -> int:
    return secrets.randbelow(SEED_MAX + 1)


def generative_ops(state: EditState) -> list[GenerativeBase]:
    return [op for op in state.all_operations() if isinstance(op, GenerativeBase)]


def stamp(state: EditState, model_for: Callable[[str], str] | None = None) -> EditState:
    """The state with a seed and a model recorded on every generative operation that lacks
    them. `model_for` names the backend that will run a task."""

    def missing(op: GenerativeBase) -> bool:
        return op.seed is None or (model_for is not None and not op.model)

    if not any(missing(op) for op in generative_ops(state)):
        return state
    stamped = state.model_copy(deep=True)
    for op in generative_ops(stamped):
        if op.seed is None:
            op.seed = new_seed()
        if model_for is not None and not op.model:
            op.model = model_for(task_for(op))
    return stamped


def job_params(op: GenerativeBase) -> dict[str, Any]:
    """What the model worker needs to generate this operation's result."""
    params: dict[str, Any] = {"seed": op.seed or 0}
    prompt = getattr(op, "prompt", None)
    if prompt is not None:
        params["prompt"] = prompt
    if op.model and op.model != "classical":
        params[PREFER] = op.model
    return params


NOT_IN_KEY = {"id", "harmonize", "amount"}
"""Fields that do not change what a model generates (they apply afterwards, or not at all)."""


def cache_identity(op: OpBase) -> dict[str, Any]:
    """The parts of an operation a generated result depends on (not its id)."""
    return op.model_dump(exclude=NOT_IN_KEY)
