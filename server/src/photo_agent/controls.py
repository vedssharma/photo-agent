"""Describe each operation's parameters for the manual controls in the web app.

The web app builds a slider or a picker for every parameter of every operation from this,
so a new operation gets controls without frontend changes.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

from photo_agent.operations import GEOMETRY_TYPES, OPERATIONS_BY_NAME, OpBase

Group = Literal["light", "color", "detail", "framing", "finishing", "retouch"]

GROUPS: dict[str, Group] = {
    "exposure": "light",
    "contrast": "light",
    "highlights": "light",
    "shadows": "light",
    "whites": "light",
    "blacks": "light",
    "white_balance": "color",
    "vibrance": "color",
    "saturation": "color",
    "hsl": "color",
    "sharpen": "detail",
    "noise_reduction": "detail",
    "clarity": "detail",
    "dehaze": "detail",
    "crop": "framing",
    "rotate": "framing",
    "straighten": "framing",
    "flip": "framing",
    "vignette": "finishing",
    "grain": "finishing",
    "tone_curve": "finishing",
    "remove": "retouch",
}


class ParamSpec(BaseModel):
    name: str
    label: str
    kind: Literal["number", "choice", "curve"]
    description: str = ""
    min: float | None = None
    max: float | None = None
    step: float | None = None
    choices: list[str | int] | None = None
    default: Any = None
    """A neutral starting value for a newly added operation."""


class OperationSpec(BaseModel):
    op: str
    label: str
    description: str
    group: Group
    framing: bool
    """Whether this is a crop/rotation/flip, which goes in the framing rather than a layer."""
    params: list[ParamSpec]


def _label(name: str) -> str:
    return name.replace("_", " ").capitalize()


def _step(lo: float, hi: float) -> float:
    span = hi - lo
    if span <= 1:
        return 0.01
    if span <= 10:
        return 0.05
    if span <= 90:
        return 0.1
    return 1


def _param(name: str, prop: dict[str, Any], required: bool) -> ParamSpec:
    description = str(prop.get("description", ""))
    if "enum" in prop or "const" in prop:
        choices = prop.get("enum") or [prop["const"]]
        default = prop.get("default", choices[0])
        return ParamSpec(
            name=name,
            label=_label(name),
            kind="choice",
            description=description,
            choices=choices,
            default=default,
        )
    if prop.get("type") == "array":
        return ParamSpec(
            name=name,
            label=_label(name),
            kind="curve",
            description=description,
            default=[[0.0, 0.0], [1.0, 1.0]],
        )
    lo = float(prop.get("minimum", prop.get("exclusiveMinimum", -100)))
    hi = float(prop.get("maximum", prop.get("exclusiveMaximum", 100)))
    neutral = 0.0 if lo <= 0 <= hi else lo
    return ParamSpec(
        name=name,
        label=_label(name),
        kind="number",
        description=description,
        min=lo,
        max=hi,
        step=_step(lo, hi),
        default=prop.get("default", neutral) if not required else neutral,
    )


def _spec(name: str, cls: type[OpBase]) -> OperationSpec:
    schema = cls.model_json_schema()
    defs = schema.get("$defs", {})
    required = set(schema.get("required", []))
    params = []
    for field, prop in schema["properties"].items():
        if field in ("id", "op"):
            continue
        ref = prop.get("$ref")
        if isinstance(ref, str):
            prop = {
                **defs[ref.rsplit("/", 1)[-1]],
                **{k: v for k, v in prop.items() if k != "$ref"},
            }
        params.append(_param(field, prop, field in required))
    return OperationSpec(
        op=name,
        label=_label(name),
        description=" ".join((cls.__doc__ or "").split()),
        group=GROUPS[name],
        framing=cls in GEOMETRY_TYPES,
        params=params,
    )


def operation_specs() -> list[OperationSpec]:
    return [_spec(name, cls) for name, cls in OPERATIONS_BY_NAME.items()]
