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


SEED_MAX = 2**31 - 1
APP_FIELDS = ("model",)
"""Fields the app fills in on generative operations; tools and sliders leave them out."""


class GenerativeBase(OpBase):
    """An operation whose pixels come from a generative model. The prompt, the seed, and the
    model are recorded, so the same result can be rendered again, or varied by a new seed."""

    seed: int | None = Field(
        None,
        ge=0,
        le=SEED_MAX,
        description="The same prompt and seed give the same result; leave it out for a "
        "fresh one, or change it for a different take.",
    )
    model: str = Field(
        "", max_length=80, description="The model that generates it, recorded by the app."
    )

    def summary(self) -> str:
        title = self.op.replace("_", " ").capitalize()  # type: ignore[attr-defined]
        prompt = getattr(self, "prompt", "")
        what = f" “{prompt}”" if prompt else ""
        seed = f" (seed {self.seed})" if self.seed is not None else ""
        return f"{title}{what}{seed}"


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


class Perspective(OpBase):
    """Fix lines that converge because the camera was tilted: buildings that lean in toward
    the top, or a wall that shrinks toward one side. The narrow side is stretched out to the
    frame, so nothing is left empty."""

    op: Literal["perspective"] = "perspective"
    vertical: float = Field(
        0,
        ge=-100,
        le=100,
        description="Positive widens the top, for verticals that lean in toward the top "
        "(camera tilted up); negative widens the bottom.",
    )
    horizontal: float = Field(
        0,
        ge=-100,
        le=100,
        description="Positive widens the right side, for lines that converge toward the "
        "right; negative widens the left.",
    )


class LensCorrection(OpBase):
    """Undo lens distortion: straight lines that bow outward (barrel, wide angle) or inward
    (pincushion, telephoto). Scales up just enough that no edge is left empty."""

    op: Literal["lens_correction"] = "lens_correction"
    distortion: float = Field(
        0,
        ge=-100,
        le=100,
        description="Positive straightens lines that bow outward (barrel); negative "
        "straightens lines that bow inward (pincushion).",
    )


ExpandAspect = Literal["free", "1:1", "4:5", "5:4", "3:4", "4:3", "2:3", "3:2", "9:16", "16:9"]
Extension = Annotated[float, Field(ge=0, le=1.5)]


class Expand(GenerativeBase):
    """Expand the canvas: extend the photo past its edges, with new surroundings painted by
    a generative model to match (outpainting). Use it to turn a vertical photo into a
    landscape one, or to give a tight shot more room, without cropping anything away. Masks
    and later crops refer to the expanded frame."""

    op: Literal["expand"] = "expand"
    aspect: ExpandAspect = Field(
        "free",
        description="Grow the frame to this width:height, centered on the photo; "
        '"free" adds the side amounts below instead.',
    )
    left: Extension = Field(0, description="Add this fraction of the width on the left.")
    right: Extension = Field(0, description="Add this fraction of the width on the right.")
    top: Extension = Field(0, description="Add this fraction of the height on top.")
    bottom: Extension = Field(0, description="Add this fraction of the height at the bottom.")
    prompt: str = Field(
        "",
        max_length=400,
        description="What the new area should show; empty continues the scene naturally.",
    )

    def summary(self) -> str:
        if self.aspect != "free":
            size = f"to {self.aspect}"
        else:
            sides = {"left": self.left, "right": self.right, "top": self.top}
            sides["bottom"] = self.bottom
            size = ", ".join(f"{k} +{v:.0%}" for k, v in sides.items() if v) or "by nothing"
        what = f" with “{self.prompt}”" if self.prompt else ""
        return f"Expand canvas {size}{what}"


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


# Retouching


class Remove(OpBase):
    """Remove what the layer's mask selects (a person, power lines, a sign) and fill the gap
    with plausible surroundings, using an AI inpainting model. Needs a mask; it is the only
    operation in its layer. Removal layers apply first, before any adjustment layer."""

    op: Literal["remove"] = "remove"
    grow: Strength = Field(
        20, description="How far past the selection's edge to fill, so outlines and halos go."
    )


# Generative


class Generate(GenerativeBase):
    """Generative fill: paint new content where the layer's mask selects, described in
    words ("a potted plant", "a sunset sky", "calm water"). The model blends it into the
    photo's light and perspective. Needs a mask; it is the only operation in its layer, and
    like removals it applies before any adjustment layer."""

    op: Literal["generate"] = "generate"
    prompt: str = Field(
        min_length=1, max_length=400, description="What to put there, in plain words."
    )
    grow: Strength = Field(
        10, description="How far past the selection's edge to repaint, for a seamless blend."
    )


class ReplaceBackground(GenerativeBase):
    """Replace the background with a new scene described in words ("a sunlit beach",
    "a softly lit studio, pale gray"), painted around the subject so its perspective fits,
    and match the subject's light and color to it. The layer's mask selects what to
    replace; without one, everything but the main subject. The only operation in its layer;
    applies before adjustments."""

    op: Literal["replace_background"] = "replace_background"
    prompt: str = Field(min_length=1, max_length=400, description="The new background.")
    harmonize: Strength = Field(
        50, description="How much to match the subject's light and color to the new scene."
    )


class SmoothSkin(OpBase):
    """Soften skin while keeping its natural texture (pores, fine lines stay, blotches and
    uneven tone go). Use on a layer masked to skin; subtle amounts (20-40) look natural."""

    op: Literal["smooth_skin"] = "smooth_skin"
    amount: Strength = Field(35, description="How much to smooth.")
    texture: Strength = Field(60, description="How much fine skin texture to keep.")


class HealBlemishes(OpBase):
    """Find small spots (blemishes, pimples, dust) and heal them from the skin around them.
    Use on a layer masked to skin."""

    op: Literal["heal_blemishes"] = "heal_blemishes"
    amount: Strength = Field(50, description="How readily spots are healed; higher finds more.")
    size: Strength = Field(40, description="Largest spot to heal; higher heals bigger spots.")


class ToneCurve(OpBase):
    """Remap tones with a curve through control points (input, output), both 0..1.

    The curve always passes through its points in order of input. An S-shape adds contrast;
    lifting the (0, y) point gives faded blacks.
    """

    op: Literal["tone_curve"] = "tone_curve"
    points: list[Annotated[list[Unit], Field(min_length=2, max_length=2)]] = Field(
        min_length=2, max_length=16, description="[input, output] pairs."
    )
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
    | Perspective
    | LensCorrection
    | Vignette
    | Grain
    | ToneCurve
    | Remove
    | SmoothSkin
    | HealBlemishes
    | Generate
    | Expand
    | ReplaceBackground,
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
    Perspective,
    LensCorrection,
    Vignette,
    Grain,
    ToneCurve,
    Remove,
    SmoothSkin,
    HealBlemishes,
    Generate,
    Expand,
    ReplaceBackground,
)

OperationAdapter: TypeAdapter[Operation] = TypeAdapter(Operation)

GEOMETRY_TYPES: tuple[type[OpBase], ...] = (
    Crop,
    Rotate,
    Straighten,
    Flip,
    Perspective,
    LensCorrection,
    Expand,
)
"""Operations that change framing rather than look."""

FramingOperation = Annotated[
    Crop | Rotate | Straighten | Flip | Perspective | LensCorrection | Expand,
    Field(discriminator="op"),
]
"""Crop, rotation, flips, perspective and lens fixes, and canvas expansion. They apply to
the whole photo, before any layer."""

AdjustmentOperation = Annotated[
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
    | Vignette
    | Grain
    | ToneCurve
    | Remove
    | SmoothSkin
    | HealBlemishes
    | Generate
    | ReplaceBackground,
    Field(discriminator="op"),
]
"""Everything that changes the look rather than the framing; these live in layers."""

CONTENT_TYPES: tuple[type[OpBase], ...] = (Remove, Generate, ReplaceBackground)
"""Operations that change what is in the photo rather than how it looks. Each is the only
operation in its layer, and content layers render before every adjustment layer."""

MASKED_TYPES: tuple[type[OpBase], ...] = (Remove, Generate)
"""Content operations that only make sense where a mask points (the rest default to a
sensible region, or the whole photo)."""


def op_name(cls: type[OpBase]) -> str:
    default = cls.model_fields["op"].default
    assert isinstance(default, str)
    return default


OPERATIONS_BY_NAME: dict[str, type[OpBase]] = {op_name(cls): cls for cls in OPERATION_TYPES}
