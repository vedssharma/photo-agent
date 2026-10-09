"""Suggestions: on open, a few edit directions for the photo to pick from.

Claude looks at the photo and proposes two to four distinct directions ("Bright and airy",
"Moody film", "Classic black and white"), each as a handful of slider values and, if one
fits, a built-in look. Each direction becomes ordinary layers, so the app can render a
thumbnail of it, and picking one adds those layers as one step that can be refined in chat
or by hand. Without an API key a built-in set, tuned by measuring the photo, stands in.
"""

from __future__ import annotations

import math
import uuid
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

from photo_agent import diagnostics, looks
from photo_agent import operations as ops
from photo_agent.advisor import Advisor
from photo_agent.imaging import Array
from photo_agent.layers import EditState, Layer, new_layer_id
from photo_agent.render import luma

MAX_SUGGESTIONS = 4
THUMBNAIL_LONG_EDGE = 480

SLIDERS: dict[str, tuple[float, float]] = {
    "exposure": (-2, 2),
    "contrast": (-100, 100),
    "highlights": (-100, 100),
    "shadows": (-100, 100),
    "whites": (-100, 100),
    "blacks": (-100, 100),
    "temperature": (-100, 100),
    "tint": (-100, 100),
    "vibrance": (-100, 100),
    "saturation": (-100, 100),
    "clarity": (-100, 100),
    "dehaze": (-100, 100),
    "vignette": (-100, 100),
    "grain": (0, 100),
}
"""The sliders a direction can set, with their ranges. 0 leaves one alone."""

Sliders = dict[str, float]


class Direction(BaseModel):
    """One way the photo could go, as Claude describes it."""

    title: str
    description: str
    look: str = "none"
    look_strength: float = 100
    sliders: Sliders = Field(default_factory=dict)


class Suggestion(BaseModel):
    id: str
    title: str = Field(description="A short name, like “Moody film”.")
    description: str = Field(description="One friendly sentence on what it does.")
    layers: list[Layer] = Field(description="The layers picking it adds on top.")


Source = Literal["claude", "built-in"]


class SuggestionSet(BaseModel):
    revision: str
    """The edit state the suggestions were made for."""
    source: Source
    suggestions: list[Suggestion]


SYSTEM = """You are the photo editor inside photo-agent, for people who are not editing \
experts. A person just opened the photo shown. Propose two to four distinct directions \
the edit could go, each something a good photographer would actually do with this photo, \
and different enough from each other to be worth choosing between: for example a natural \
fix that only corrects what is off, a mood that suits the subject, and a bolder creative \
take. Give each a short title (two to four words) and one friendly sentence in plain words.

Each direction is a set of slider values (0 leaves a slider alone; exposure is in stops, \
-2 to 2; grain is 0 to 100; the rest are -100 to 100, where typical amounts are 10 to 40) \
and optionally one of the built-in looks with a strength (0 to 100). Keep skin natural, \
avoid blown highlights and garish color, and err on the side of subtle.

Built-in looks: """ + "; ".join(
    f"{look.id}: {look.name}, {look.description}" for look in looks.LOOKS
)

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "directions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "look": {"type": "string", "enum": ["none", *(lk.id for lk in looks.LOOKS)]},
                    "look_strength": {"type": "number"},
                    "sliders": {
                        "type": "object",
                        "properties": {name: {"type": "number"} for name in SLIDERS},
                        "required": list(SLIDERS),
                        "additionalProperties": False,
                    },
                },
                "required": ["title", "description", "look", "look_strength", "sliders"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["directions"],
    "additionalProperties": False,
}


def _clamp(name: str, value: float) -> float:
    low, high = SLIDERS[name]
    return round(min(high, max(low, float(value))), 2)


def slider_operations(sliders: Sliders) -> list[ops.OpBase]:
    """The operations that set these sliders, skipping the ones left at 0."""
    s = {name: _clamp(name, v) for name, v in sliders.items() if name in SLIDERS}
    out: list[ops.OpBase] = []
    if s.get("exposure"):
        out.append(ops.Exposure(stops=s["exposure"]))
    if s.get("temperature") or s.get("tint"):
        out.append(ops.WhiteBalance(temperature=s.get("temperature", 0), tint=s.get("tint", 0)))
    singles: dict[str, type[ops.OpBase]] = {
        "contrast": ops.Contrast,
        "highlights": ops.Highlights,
        "shadows": ops.Shadows,
        "whites": ops.Whites,
        "blacks": ops.Blacks,
        "vibrance": ops.Vibrance,
        "saturation": ops.Saturation,
        "clarity": ops.Clarity,
        "dehaze": ops.Dehaze,
        "vignette": ops.Vignette,
        "grain": ops.Grain,
    }
    for name, cls in singles.items():
        if s.get(name):
            out.append(cls(amount=s[name]))  # type: ignore[call-arg]
    return out


def to_suggestion(direction: Direction) -> Suggestion | None:
    """The layers for a direction; None if it would change nothing."""
    layers: list[Layer] = []
    title = " ".join(direction.title.split())[:80] or "Suggestion"
    if direction.look != "none":
        try:
            look = looks.get(direction.look)
        except looks.LookNotFoundError:
            look = None
        if look is not None:
            strength = min(100.0, max(0.0, float(direction.look_strength)))
            if strength > 0:
                layers.append(looks.layer(look, strength))
    adjustments = slider_operations(direction.sliders)
    if adjustments:
        name = title if not layers else f"{title}: tuning"
        layers.append(
            Layer.model_validate(
                {
                    "id": new_layer_id(),
                    "name": name[:80],
                    "operations": [op.model_dump() for op in adjustments],
                }
            )
        )
    if not layers:
        return None
    return Suggestion(
        id=uuid.uuid4().hex[:8],
        title=title,
        description=" ".join(direction.description.split())[:300],
        layers=layers,
    )


def built_in_directions(pixels: Array) -> list[Direction]:
    """Directions worked out from measurements, for when Claude is not available."""
    m = diagnostics.measure(pixels)
    spread = float(np.std(luma(np.clip(pixels, 0, 1))))
    fix: Sliders = {"vibrance": 15}
    if m.mean_brightness > 0:
        stops = math.log2(0.46 / max(m.mean_brightness, 0.02)) * 0.6
        if abs(stops) >= 0.1:
            fix["exposure"] = max(-1.0, min(1.0, round(stops, 2)))
    if m.clipped_highlights > 0.02:
        fix["highlights"] = -35
    if m.crushed_shadows > 0.02 or m.mean_brightness < 0.3:
        fix["shadows"] = 25
    if spread < 0.18:
        fix["contrast"] = 15
    return [
        Direction(
            title="Natural fix",
            description="Balances the brightness and gently lifts the color, nothing more.",
            sliders=fix,
        ),
        Direction(
            title="Warm and soft",
            description="Golden, gentle light with softer contrast, flattering for people.",
            sliders={"temperature": 15, "contrast": -10, "highlights": -15, "shadows": 10},
        ),
        Direction(
            title="Bold and punchy",
            description="Deeper contrast, crisper detail, and richer color that stands out.",
            sliders={"contrast": 25, "clarity": 15, "vibrance": 25, "vignette": -15},
        ),
        Direction(
            title="Classic black and white",
            description="Timeless monochrome with deep blacks.",
            look="noir",
            look_strength=80,
        ),
    ]


async def ask_claude(
    advisor: Advisor, image_block: dict[str, Any], measurements: diagnostics.Measurements
) -> list[Direction]:
    answer = await advisor.ask(
        system=SYSTEM,
        content=[
            image_block,  # type: ignore[list-item]
            {
                "type": "text",
                "text": "Measurements of the unedited photo: "
                f"{measurements.clipped_highlights:.1%} blown-out highlights, "
                f"{measurements.crushed_shadows:.1%} pure black, "
                f"{measurements.oversaturated:.1%} maximally saturated, "
                f"average brightness {measurements.mean_brightness:.2f} (0 to 1).\n\n"
                "Propose the directions.",
            },
        ],
        schema=SCHEMA,
        tier="deep",
    )
    raw = answer.get("directions")
    if not isinstance(raw, list):
        return []
    found = []
    for item in raw:
        try:
            found.append(Direction.model_validate(item))
        except ValueError:
            continue
    return found


def suggestions_from(directions: list[Direction]) -> list[Suggestion]:
    made = [s for s in map(to_suggestion, directions) if s is not None]
    return made[:MAX_SUGGESTIONS]


def apply(suggestion: Suggestion, state: EditState) -> EditState:
    """The state with the suggestion's layers added on top, with fresh ids."""
    added = [
        layer.model_copy(
            update={
                "id": new_layer_id(),
                "operations": [
                    op.model_copy(update={"id": ops.new_op_id()}) for op in layer.operations
                ],
            }
        )
        for layer in suggestion.layers
    ]
    return state.model_copy(update={"layers": [*state.layers, *added]}, deep=True)
