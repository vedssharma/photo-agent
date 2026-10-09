"""Portrait retouching: a few masked layers that make a portrait look polished, not fake.

Each part is its own layer with a face-part mask found by the face-parsing model, so the
person can hide or fade any of them. Defaults are subtle on purpose.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from photo_agent import operations as ops
from photo_agent.layers import Layer, new_layer_id
from photo_agent.masks import SemanticMask, SemanticTarget


class Retouch(BaseModel):
    """How much of each retouch to do; 0 leaves that part out."""

    model_config = ConfigDict(extra="forbid")

    smooth_skin: float = Field(35, ge=0, le=100, description="Soften skin, keeping texture.")
    remove_blemishes: bool = Field(True, description="Heal small spots on the skin.")
    brighten_eyes: float = Field(25, ge=0, le=100, description="Brighten and clarify eyes.")
    whiten_teeth: float = Field(25, ge=0, le=100, description="Take the yellow out of teeth.")


def retouch_layers(options: Retouch) -> list[Layer]:
    """The layers for a portrait retouch, bottom first."""
    layers = []

    def layer(name: str, target: SemanticTarget, operations: list[ops.OpBase]) -> None:
        layers.append(
            Layer.model_validate(
                {
                    "id": new_layer_id(),
                    "name": name,
                    "mask": SemanticMask(target=target),
                    "operations": operations,
                }
            )
        )

    skin: list[ops.OpBase] = []
    if options.remove_blemishes:
        skin.append(ops.HealBlemishes())
    if options.smooth_skin > 0:
        skin.append(ops.SmoothSkin(amount=options.smooth_skin))
    if skin:
        layer("Smooth skin" if options.smooth_skin > 0 else "Heal blemishes", "skin", skin)
    if options.brighten_eyes > 0:
        a = options.brighten_eyes / 100
        layer(
            "Brighten eyes",
            "eyes",
            [ops.Exposure(stops=round(0.6 * a, 2)), ops.Clarity(amount=round(30 * a))],
        )
    if options.whiten_teeth > 0:
        a = options.whiten_teeth / 100
        layer(
            "Whiten teeth",
            "teeth",
            [
                ops.HSL(band="yellow", saturation=round(-80 * a), luminance=round(30 * a)),
                ops.Exposure(stops=round(0.3 * a, 2)),
            ],
        )
    return layers
