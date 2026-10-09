"""Looks: named styles ("1970s film", "teal and orange") made of ordinary adjustments.

A look adds one layer of color grading, tone, and finishing operations, so it is as
editable as anything else: fade it with the layer's opacity, tweak any slider, or hide it.
Looks change color and tone only; redrawing a photo in a new medium (a painting, anime) is
the `restyle` operation, which needs a generative model.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from photo_agent import operations as ops
from photo_agent.layers import EditState, Layer, new_layer_id


class Look(BaseModel):
    id: str
    name: str
    description: str
    operations: list[ops.AdjustmentOperation] = Field(min_length=1)


def _look(id: str, name: str, description: str, *operations: ops.OpBase) -> Look:
    return Look.model_validate(
        {
            "id": id,
            "name": name,
            "description": description,
            "operations": [op.model_dump() for op in operations],
        }
    )


LOOKS: list[Look] = [
    _look(
        "film-70s",
        "1970s film",
        "Warm, faded, and grainy, like a print from a family album.",
        ops.ToneCurve(points=[[0, 0.08], [0.25, 0.27], [0.75, 0.76], [1, 0.94]]),
        ops.ColorGrade(
            shadows_hue=170,
            shadows=25,
            midtones_hue=45,
            midtones=30,
            highlights_hue=40,
            highlights=35,
        ),
        ops.Saturation(amount=-15),
        ops.Grain(amount=30, size=40),
        ops.Vignette(amount=-15),
    ),
    _look(
        "teal-orange",
        "Teal and orange",
        "The blockbuster grade: teal shadows, warm skin and highlights.",
        ops.ColorGrade(shadows_hue=195, shadows=55, highlights_hue=30, highlights=45),
        ops.Contrast(amount=15),
        ops.Vibrance(amount=10),
    ),
    _look(
        "noir",
        "Noir",
        "Black and white with deep blacks and hard contrast.",
        ops.Saturation(amount=-100),
        ops.Contrast(amount=40),
        ops.Blacks(amount=-20),
        ops.Clarity(amount=20),
        ops.Vignette(amount=-30),
        ops.Grain(amount=15),
    ),
    _look(
        "soft-portrait-film",
        "Soft portrait film",
        "Gentle contrast, creamy highlights, and flattering warm skin.",
        ops.Contrast(amount=-12),
        ops.Highlights(amount=-20),
        ops.ColorGrade(midtones_hue=25, midtones=20, highlights_hue=340, highlights=12),
        ops.Vibrance(amount=-5),
        ops.Grain(amount=10),
    ),
    _look(
        "pastel-storybook",
        "Pastel storybook",
        "Soft pastel colors, pink highlights, and lifted blacks, like a whimsical film set.",
        ops.ToneCurve(points=[[0, 0.1], [0.5, 0.53], [1, 0.97]]),
        ops.ColorGrade(midtones_hue=40, midtones=25, highlights_hue=350, highlights=30),
        ops.HSL(band="yellow", saturation=20),
        ops.HSL(band="red", saturation=-10, luminance=10),
        ops.Contrast(amount=-10),
    ),
    _look(
        "cinematic-cool",
        "Cool cinematic",
        "Muted color with blue-green shadows, for a moody, modern film.",
        ops.ColorGrade(shadows_hue=200, shadows=45, midtones_hue=190, midtones=10),
        ops.Saturation(amount=-25),
        ops.Contrast(amount=20),
        ops.Vignette(amount=-20),
    ),
    _look(
        "golden-hour",
        "Golden hour",
        "Late-afternoon warmth: golden light and glowing highlights.",
        ops.WhiteBalance(temperature=25),
        ops.ColorGrade(highlights_hue=38, highlights=40, midtones_hue=30, midtones=15),
        ops.Vibrance(amount=15),
        ops.Shadows(amount=10),
    ),
    _look(
        "faded-matte",
        "Faded matte",
        "Flat, airy, and low contrast, with milky blacks.",
        ops.ToneCurve(points=[[0, 0.12], [0.5, 0.52], [1, 0.95]]),
        ops.Contrast(amount=-15),
        ops.Saturation(amount=-15),
    ),
    _look(
        "bleach-bypass",
        "Bleach bypass",
        "Desaturated and gritty, with silvery highlights.",
        ops.Saturation(amount=-50),
        ops.Contrast(amount=35),
        ops.Clarity(amount=25),
        ops.Highlights(amount=-10),
    ),
    _look(
        "cross-process",
        "Cross-processed",
        "Punchy and off-kilter: yellow-green highlights, blue shadows.",
        ops.ColorGrade(shadows_hue=230, shadows=40, highlights_hue=75, highlights=45),
        ops.Contrast(amount=20),
        ops.Saturation(amount=15),
    ),
]

LOOKS_BY_ID = {look.id: look for look in LOOKS}


class LookNotFoundError(KeyError):
    pass


def get(look_id: str) -> Look:
    try:
        return LOOKS_BY_ID[look_id]
    except KeyError:
        raise LookNotFoundError(look_id) from None


def layer(look: Look, strength: float = 100) -> Layer:
    """A new layer with the look's operations, at `strength` (0..100) opacity."""
    operations = [op.model_copy(update={"id": ops.new_op_id()}) for op in look.operations]
    return Layer(id=new_layer_id(), name=look.name, opacity=strength, operations=operations)


def apply(look: Look, state: EditState, strength: float = 100) -> EditState:
    """The state with the look's layer added on top."""
    return state.model_copy(update={"layers": [*state.layers, layer(look, strength)]}, deep=True)
