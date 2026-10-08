"""The core operation toolbox, as data.

Each class here is one kind of edit the agent can make. An operation is only a name and
parameters; `photo_agent.render` turns a list of them into pixels. Docstrings and field
descriptions double as the Claude tool definitions, so they are written for the agent.

Slider-style amounts use a -100..100 scale like consumer photo apps, where 0 is "no change".
"""

from __future__ import annotations

import uuid
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

Amount = Annotated[float, Field(ge=-100, le=100)]
Strength = Annotated[float, Field(ge=0, le=100)]
Unit = Annotated[float, Field(ge=0, le=1)]


def new_op_id() -> str:
    return uuid.uuid4().hex[:8]


class OpBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=new_op_id, description="Stable id of this operation.")

    def summary(self) -> str:
        """A short human-readable description, e.g. 'Exposure +0.4'."""
        params = self.model_dump(exclude={"id", "op"})
        parts = [f"{k} {_fmt(v)}" for k, v in params.items()]
        title = self.op.replace("_", " ").capitalize()  # type: ignore[attr-defined]
        return f"{title} ({', '.join(parts)})" if parts else title


def _fmt(v: object) -> str:
    if isinstance(v, float):
        return f"{v:+g}"
    return str(v)


# Light


class Exposure(OpBase):
    """Brighten or darken the whole photo, like changing the camera exposure."""

    op: Literal["exposure"] = "exposure"
    stops: float = Field(ge=-5, le=5, description="Exposure change in stops; +1 doubles light.")


class Contrast(OpBase):
    """Increase or decrease overall contrast around the midtones."""

    op: Literal["contrast"] = "contrast"
    amount: Amount = Field(description="Positive adds punch; negative flattens.")


class Highlights(OpBase):
    """Recover (negative) or boost (positive) the brightest areas without touching shadows."""

    op: Literal["highlights"] = "highlights"
    amount: Amount


class Shadows(OpBase):
    """Lift (positive) or deepen (negative) the darkest areas without touching highlights."""

    op: Literal["shadows"] = "shadows"
    amount: Amount


class Whites(OpBase):
    """Move the white point: positive makes the brightest tones brighter, negative dims them."""

    op: Literal["whites"] = "whites"
    amount: Amount


class Blacks(OpBase):
    """Move the black point: negative makes blacks deeper, positive lifts them (matte look)."""

    op: Literal["blacks"] = "blacks"
    amount: Amount


# Color


class WhiteBalance(OpBase):
    """Shift color temperature and tint to fix a color cast or set a mood."""

    op: Literal["white_balance"] = "white_balance"
    temperature: Amount = Field(0, description="Positive is warmer (yellow), negative cooler.")
    tint: Amount = Field(0, description="Positive is more magenta, negative more green.")


class Vibrance(OpBase):
    """Boost or reduce saturation mostly in muted colors, protecting skin and saturated areas."""

    op: Literal["vibrance"] = "vibrance"
    amount: Amount


class Saturation(OpBase):
    """Uniformly increase or decrease color intensity; -100 makes black and white."""

    op: Literal["saturation"] = "saturation"
    amount: Amount


ColorBand = Literal["red", "orange", "yellow", "green", "aqua", "blue", "purple", "magenta"]


class HSL(OpBase):
    """Adjust hue, saturation, and luminance of one color band (e.g. deepen blue skies)."""

    op: Literal["hsl"] = "hsl"
    band: ColorBand
    hue: Amount = Field(0, description="Shift the band's hue toward its neighbors.")
    saturation: Amount = 0
    luminance: Amount = 0


# Detail


class Sharpen(OpBase):
    """Sharpen edges and fine detail."""

    op: Literal["sharpen"] = "sharpen"
    amount: float = Field(ge=0, le=150, description="Strength; 25-50 is typical.")
    radius: float = Field(1.0, ge=0.5, le=3, description="Detail size in full-res pixels.")


class NoiseReduction(OpBase):
    """Smooth out grain and color speckles, typical of low-light photos."""

    op: Literal["noise_reduction"] = "noise_reduction"
    luminance: Strength = Field(0, description="Smooths grainy brightness noise.")
    color: Strength = Field(0, description="Removes colored speckles.")


class Clarity(OpBase):
    """Add (positive) or soften (negative) midtone local contrast and texture."""

    op: Literal["clarity"] = "clarity"
    amount: Amount


class Dehaze(OpBase):
    """Cut through haze or fog (positive), or add atmosphere (negative)."""

    op: Literal["dehaze"] = "dehaze"
    amount: Amount


# Geometry

AspectRatio = Literal[
    "free", "original", "1:1", "4:5", "5:4", "3:4", "4:3", "2:3", "3:2", "9:16", "16:9"
]


class Crop(OpBase):
    """Crop to a region of the current frame, optionally locked to an aspect ratio.

    The box is given as fractions of the current width and height (0 = left/top edge,
    1 = right/bottom edge). With an aspect ratio, the result is the largest rectangle of that
    ratio centered inside the box, so a full-frame box plus "4:5" is a centered Instagram crop.
    """

    op: Literal["crop"] = "crop"
    left: Unit = 0
    top: Unit = 0
    right: Unit = 1
    bottom: Unit = 1
    aspect: AspectRatio = Field("free", description="Aspect ratio as width:height.")

    @model_validator(mode="after")
    def _check_box(self) -> Crop:
        if self.right - self.left < 0.01 or self.bottom - self.top < 0.01:
            raise ValueError("crop box must have right > left and bottom > top")
        return self


class Rotate(OpBase):
    """Rotate the photo clockwise by a quarter, half, or three-quarter turn."""

    op: Literal["rotate"] = "rotate"
    degrees: Literal[90, 180, 270]


class Straighten(OpBase):
    """Rotate by a small angle to level a tilted horizon, cropping away the empty corners."""

    op: Literal["straighten"] = "straighten"
    angle: float = Field(ge=-45, le=45, description="Degrees; positive turns clockwise.")


class Flip(OpBase):
    """Mirror the photo."""

    op: Literal["flip"] = "flip"
    axis: Literal["horizontal", "vertical"] = "horizontal"


# Finishing


class Vignette(OpBase):
    """Darken (negative) or lighten (positive) the edges to draw the eye inward."""

    op: Literal["vignette"] = "vignette"
    amount: Amount
    midpoint: Strength = Field(50, description="How far in the effect reaches; lower is wider.")


class Grain(OpBase):
    """Add film-like grain."""

    op: Literal["grain"] = "grain"
    amount: Strength
    size: Strength = Field(25, description="Grain size; larger is coarser.")


class ToneCurve(OpBase):
    """Remap tones with a curve through control points (input, output), both 0..1.

    The curve always passes through its points in order of input. An S-shape adds contrast;
    lifting the (0, y) point gives faded blacks.
    """

    op: Literal["tone_curve"] = "tone_curve"
    points: list[tuple[Unit, Unit]] = Field(min_length=2, max_length=16)
    channel: Literal["rgb", "red", "green", "blue"] = "rgb"


Operation = Annotated[
    Exposure
    | Contrast
    | Highlights
    | Shadows
    | Whites
    | Blacks
    | WhiteBalance
    | Vibrance
    | Saturation
    | HSL
    | Sharpen
    | NoiseReduction
    | Clarity
    | Dehaze
    | Crop
    | Rotate
    | Straighten
    | Flip
    | Vignette
    | Grain
    | ToneCurve,
    Field(discriminator="op"),
]

OPERATION_TYPES: tuple[type[OpBase], ...] = (
    Exposure,
    Contrast,
    Highlights,
    Shadows,
    Whites,
    Blacks,
    WhiteBalance,
    Vibrance,
    Saturation,
    HSL,
    Sharpen,
    NoiseReduction,
    Clarity,
    Dehaze,
    Crop,
    Rotate,
    Straighten,
    Flip,
    Vignette,
    Grain,
    ToneCurve,
)

OperationAdapter: TypeAdapter[Operation] = TypeAdapter(Operation)


def op_name(cls: type[OpBase]) -> str:
    default = cls.model_fields["op"].default
    assert isinstance(default, str)
    return default


OPERATIONS_BY_NAME: dict[str, type[OpBase]] = {op_name(cls): cls for cls in OPERATION_TYPES}
