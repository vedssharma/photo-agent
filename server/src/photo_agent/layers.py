"""Edit state: the framing of the photo plus a stack of adjustment layers.

Framing (crop, rotation, flips) applies first, to the whole photo. Then removal layers (which
fill in what their mask selects) clean up the photo, and each visible adjustment layer,
bottom to top, applies its operations to the result so far and is blended back in by its
blend mode, opacity, and mask. That lets each change stay separate: it can be hidden, faded, or
removed without touching the others.
"""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Iterator, Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from photo_agent.masks import Mask
from photo_agent.operations import (
    GEOMETRY_TYPES,
    AdjustmentOperation,
    FramingOperation,
    OpBase,
    Operation,
    Remove,
)

BlendMode = Literal["normal", "multiply", "screen", "overlay", "soft_light", "color", "luminosity"]

BLEND_MODE_HELP = (
    "How the layer's result combines with the layers below: normal replaces them; "
    "luminosity changes only brightness (contrast without color shifts); color changes only "
    "color; multiply darkens; screen lightens; overlay and soft_light add contrast."
)


def new_layer_id() -> str:
    return "L" + uuid.uuid4().hex[:7]


class Layer(BaseModel):
    """A named group of adjustments with its own visibility, opacity, and blend mode."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Stable id; new layers get one from `new_layer_id`.")
    name: str = Field(min_length=1, max_length=80)
    visible: bool = True
    opacity: float = Field(100, ge=0, le=100, description="0 is no effect, 100 is full effect.")
    blend_mode: BlendMode = Field("normal", description=BLEND_MODE_HELP)
    mask: Mask | None = Field(
        None, description="Limits the layer to part of the photo; null applies it everywhere."
    )
    operations: list[AdjustmentOperation] = Field([])

    @model_validator(mode="after")
    def _removal_stands_alone(self) -> Layer:
        if self.is_removal and len(self.operations) > 1:
            raise ValueError("a remove operation must be the only operation in its layer")
        return self

    @property
    def is_removal(self) -> bool:
        """Removes what its mask selects. Removal layers render before all others."""
        return any(isinstance(op, Remove) for op in self.operations)


class EditState(BaseModel):
    """Everything needed to render the photo from the original."""

    model_config = ConfigDict(extra="forbid")

    framing: list[FramingOperation] = Field([])
    """Crop, rotation, and flips, applied in order before any layer."""
    layers: list[Layer] = Field([])
    """Adjustment layers, bottom first."""

    @classmethod
    def from_operations(cls, operations: Sequence[Operation], name: str) -> EditState:
        """Split a flat Phase 1 operation list into framing and a single layer."""
        framing = [op for op in operations if isinstance(op, GEOMETRY_TYPES)]
        looks = [op for op in operations if not isinstance(op, GEOMETRY_TYPES)]
        layer = {"id": new_layer_id(), "name": name[:80] or "Edits", "operations": looks}
        return cls.model_validate({"framing": framing, "layers": [layer] if looks else []})

    def all_operations(self) -> Iterator[OpBase]:
        yield from self.framing
        for layer in self.layers:
            yield from layer.operations

    def layer(self, layer_id: str) -> Layer:
        for layer in self.layers:
            if layer.id == layer_id:
                return layer
        raise KeyError(layer_id)

    @property
    def fingerprint(self) -> str:
        """A short hash that changes whenever the rendered result could (renaming a layer,
        for example, does not change it)."""
        payload = self.model_dump_json(exclude={"layers": {"__all__": {"name"}}})
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    @property
    def is_empty(self) -> bool:
        return not self.framing and not self.layers
