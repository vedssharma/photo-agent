"""Agent service: Claude turns a chat request into edit operations.

Claude never touches pixels. Each operation in the toolbox is a tool; calling one adds a
step to the document's operation list. Claude sees the current preview and the operation
list with every request, so it can reason about the photo and about what it already did.
"""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import Awaitable, Callable, Sequence
from typing import Annotated, Any, Literal, Protocol

from anthropic import AsyncAnthropic
from anthropic.types.beta import (
    BetaImageBlockParam,
    BetaMessage,
    BetaMessageParam,
    BetaToolParam,
)
from pydantic import BaseModel, Field, ValidationError

from photo_agent import diagnostics, imaging
from photo_agent.generative import stamp, task_for
from photo_agent.geometry import AutoStraighten, Framed, auto_level
from photo_agent.graph import ChatEntry, Document, DocumentView, Step
from photo_agent.layers import BLEND_MODE_HELP, Cutout, EditState, Layer, new_layer_id
from photo_agent.masks import SemanticMask, describe_mask
from photo_agent.operations import (
    APP_FIELDS,
    CONTENT_TYPES,
    GEOMETRY_TYPES,
    MASKED_TYPES,
    OPERATIONS_BY_NAME,
    GenerativeBase,
    OpBase,
    Operation,
    OperationAdapter,
)
from photo_agent.portrait import Retouch, retouch_layers
from photo_agent.render import RenderCache, render
from photo_agent.store import DocumentStore

MAX_MODEL_CALLS = 8
"""Upper bound on model round trips in one turn, so a confused agent cannot loop forever."""
CLAUDE_IMAGE_LONG_EDGE = 1024

SYSTEM_PROMPT = """\
You are the photo editor inside photo-agent. The people you help are not editing experts; \
they describe what they want in everyday words and expect a professional-looking result.

How editing works:
- You never change pixels yourself. You edit by calling tools, and every change stays \
separate and reversible.
- The photo has framing (crop, rotate, straighten, flip), which applies first, and then a \
stack of adjustment layers, applied bottom to top. Each layer has a name, its own \
operations, an opacity, a blend mode, and can be hidden.
- Group each distinct change into its own layer, so the person can see, hide, or fade it on \
its own: call add_layer with a short name in plain words (like "Warmer tones" or "Brighter \
shadows") before the operations for that change. Operation tools add to the layer you added \
most recently in this request; framing tools always go to the framing.
- A layer can have a mask to limit it to part of the photo: a linear gradient for skies \
or foregrounds, a radial gradient for a subject or a spotlight (invert it to work on the \
surroundings), or a luminosity range to target highlights or shadows. Positions are \
fractions of the framed photo. Brush masks are painted by the person; keep them unless \
asked to change them.
- A semantic mask selects things by what they are, found by an AI model: the sky, the \
main subject, people, or parts of faces (skin, eyes, lips, teeth, hair). Prefer it over \
gradients whenever the request names a thing ("brighten just the subject", "make the sky \
more dramatic"); invert it to work on everything else (for example, the background). To \
pick out one particular thing ("the person on the left", "the red car"), use target \
object with a box around it, in fractions of the photo as you see it, a point on it when \
the box holds other things too, and a short description. The render after your edits \
shows what was selected; if it caught the wrong thing, adjust the box or points.
- To remove something (a person, power lines, a sign, a photobomber), add a layer named \
for it ("Remove the person on the left") with a mask selecting it, usually a semantic \
object with a box, then call remove. It must be the only operation in that layer. Removal \
layers apply first, before any adjustment, wherever they sit in the stack. Check the \
render: if traces remain (an outline, a shadow), raise grow or widen the selection.
- To add or replace something in the photo ("add a sunset", "replace the trash can with \
a plant", "put a boat on the water"), use generative fill: add a layer named for it with a \
mask selecting where it goes (a semantic object box around the thing to replace, a radial \
gradient or a box-shaped object selection for empty space, the sky to replace the sky), \
then call generate with a short concrete prompt describing only what to paint there \
("a terracotta pot with a leafy fern"), not the whole photo. It must be the only operation \
in its layer and applies before adjustments, like a removal. Each generative edit keeps \
its prompt and seed, so it renders the same every time; change the seed with \
update_operation for a different take. Look at the render: if it does not fit, refine the \
prompt or the mask.
- To put the subject somewhere else ("put me on a beach", "a clean studio background"), \
call replace_background with the new scene in a short prompt. It starts its own layer \
(add_layer first only to name it or to choose what to replace with a mask; without one it \
replaces everything but the main subject) and matches the subject's light and color to \
the scene (harmonize). For a plain color or transparency instead, use cut_out.
- To turn a vertical photo into a landscape one, give a tight shot more room, or fit a \
format without cropping ("make this 16:9 without cutting anything off"), call expand: it \
extends the canvas with new surroundings painted to match. Prefer an aspect ratio; use \
the side amounts to grow one side. A prompt is optional (empty continues the scene). It is \
part of the framing, so masks and later crops refer to the expanded frame.
- To remove the background or cut out the subject, call cut_out: by default it keeps \
the main subject on a transparent background (the person downloads a PNG); give a color \
such as "#ffffff" for a clean product shot, or a different mask to keep something else.
- For portraits ("make me look good", "fix my skin"), call retouch_portrait. Keep it \
subtle: people should look like themselves on a good day. Its defaults are a good start; \
tone them down for close-ups and children.
- When the horizon is tilted or a building leans ("straighten this", "fix the \
horizon", "the walls look crooked"), call auto_straighten: it measures the straight lines in \
the photo and adds a straighten and a perspective fix to the framing. If it finds nothing, \
set straighten yourself. Use lens_correction for lines that bow (wide-angle barrel).
- update_operation and remove_operation change operations already present, in any layer; \
update_layer and remove_layer change layers. Prefer adjusting what is already there over \
stacking a second operation of the same kind for the same purpose.
- With each request you get the current rendered photo and the current framing and layers. \
The person may have tweaked, hidden, or removed things by hand; respect those choices.
- Crop boxes are fractions of the frame as it is at that point in the framing list, after \
any earlier rotate, straighten, or crop. For Instagram, use 4:5 for portrait feed posts, 1:1 \
for square, 1.91:1 is not available so use 16:9 for landscape, and 9:16 for stories.

How to edit well:
- Look at the photo before deciding. Fix what the person asked for, plus anything that \
plainly undermines it (for example, warming a photo that is also underexposed).
- Err on the side of subtle. Typical amounts are 10 to 40 on the -100..100 sliders and \
-1 to +1 stop of exposure; go further only when the photo clearly needs it.
- Protect skin tones, keep highlights from blowing out, and avoid oversaturation.
- If a request needs something the tools cannot do, say so plainly and offer what you \
can do instead.
- If the request is ambiguous in a way that matters, make a reasonable choice and say which.

Checking your work: after each round of edits you get the newly rendered photo and \
measurements comparing it with the unedited original. Look at both before replying. If \
something overshot (blown-out highlights, crushed blacks, garish color, an odd cast, a crop \
that cuts off the subject), correct it with another tool call. One or two correction rounds \
are plenty; do not fiddle.

When you are done, reply in one to three short, friendly sentences in plain language: what \
you changed and why. Avoid jargon and slider numbers unless they help."""


# Events streamed to the chat panel.


class TurnStarted(BaseModel):
    type: Literal["turn_started"] = "turn_started"


class TextDelta(BaseModel):
    type: Literal["text"] = "text"
    text: str


class OperationEvent(BaseModel):
    type: Literal["operation"] = "operation"
    action: Literal["added", "updated", "removed"]
    summary: str


class TurnDone(BaseModel):
    type: Literal["done"] = "done"
    document: DocumentView


class AgentError(BaseModel):
    type: Literal["error"] = "error"
    message: str


AgentEvent = TurnStarted | TextDelta | OperationEvent | TurnDone | AgentError
ChatEvent = Annotated[AgentEvent, Field(discriminator="type")]
"""Any message the chat WebSocket sends to the browser."""


class ChatMessage(BaseModel):
    """What the browser sends over the chat WebSocket."""

    type: Literal["message"] = "message"
    text: str


Emit = Callable[[AgentEvent], Awaitable[None]]


class ModelClient(Protocol):
    """One streamed model call. Implemented by `ClaudeModel`, and by fakes in tests."""

    async def create(
        self,
        *,
        system: str,
        tools: Sequence[BetaToolParam],
        messages: Sequence[BetaMessageParam],
        on_text: Callable[[str], Awaitable[None]],
    ) -> BetaMessage: ...


class ClaudeModel:
    def __init__(self, api_key: str, model: str) -> None:
        self.client = AsyncAnthropic(api_key=api_key)
        self.model = model

    async def create(
        self,
        *,
        system: str,
        tools: Sequence[BetaToolParam],
        messages: Sequence[BetaMessageParam],
        on_text: Callable[[str], Awaitable[None]],
    ) -> BetaMessage:
        for attempt in range(3):
            try:
                async with self.client.beta.messages.stream(
                    model=self.model,
                    max_tokens=16000,
                    system=system,
                    tools=list(tools),
                    messages=list(messages),
                    thinking={"type": "adaptive"},
                    output_config={"effort": "medium"},
                    # If a safety classifier declines, retry on Anthropic's recommended model.
                    betas=["server-side-fallback-2026-07-01"],
                    fallbacks="default",
                ) as stream:
                    async for event in stream:
                        if event.type == "text":
                            await on_text(event.text)
                    return await stream.get_final_message()
            except ValueError:
                # Tool input JSON the SDK could not parse at all; re-issue the call.
                if attempt == 2:
                    raise
        raise AssertionError("unreachable")


def _referenced_defs(node: object, defs: dict[str, Any]) -> dict[str, Any]:
    """The subset of `defs` that `node` refers to, directly or through other defs."""
    found: dict[str, Any] = {}

    def walk(n: object) -> None:
        if isinstance(n, dict):
            ref = n.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/$defs/"):
                name = ref.removeprefix("#/$defs/")
                if name not in found:
                    found[name] = defs[name]
                    walk(defs[name])
            for v in n.values():
                walk(v)
        elif isinstance(n, list):
            for v in n:
                walk(v)

    walk(node)
    return found


def _plain_unions(node: Any) -> Any:
    """Turn Pydantic's discriminated unions into plain anyOf lists for tool schemas."""
    if isinstance(node, dict):
        out = {k: _plain_unions(v) for k, v in node.items() if k != "discriminator"}
        if "oneOf" in out:
            out["anyOf"] = out.pop("oneOf")
        if "anyOf" in out and len(out["anyOf"]) == 1 and isinstance(out["anyOf"][0], dict):
            inner = out.pop("anyOf")[0]
            out = {**inner, **out}
        return out
    if isinstance(node, list):
        return [_plain_unions(v) for v in node]
    return node


def tool_definitions() -> list[BetaToolParam]:
    tools: list[BetaToolParam] = []
    for name, cls in OPERATIONS_BY_NAME.items():
        schema = cls.model_json_schema()
        props = {
            k: v for k, v in schema["properties"].items() if k not in ("id", "op", *APP_FIELDS)
        }
        input_schema: dict[str, Any] = {
            "type": "object",
            "properties": props,
            "required": [r for r in schema.get("required", []) if r in props],
            "additionalProperties": False,
        }
        if "$defs" in schema:
            input_schema["$defs"] = schema["$defs"]
        tools.append(
            {
                "name": name,
                "description": " ".join((cls.__doc__ or name).split()),
                "input_schema": input_schema,
                "eager_input_streaming": True,
            }
        )
    layer_schema = Layer.model_json_schema()
    layer_props = {
        k: v for k, v in layer_schema["properties"].items() if k not in ("id", "operations")
    }
    add_layer_schema: dict[str, Any] = {
        "type": "object",
        "properties": layer_props,
        "required": ["name"],
        "additionalProperties": False,
    }
    if "$defs" in layer_schema:
        add_layer_schema["$defs"] = _referenced_defs(layer_props, layer_schema["$defs"])
        add_layer_schema = _plain_unions(add_layer_schema)
    tools.append(
        {
            "name": "add_layer",
            "description": "Start a new adjustment layer on top of the stack for one distinct "
            "change. Operation tools called after this add to it. " + BLEND_MODE_HELP,
            "input_schema": add_layer_schema,
            "eager_input_streaming": True,
        }
    )
    tools.append(
        {
            "name": "update_layer",
            "description": "Change a layer's name, visibility, opacity, blend mode, or mask, "
            "by id. "
            "Only the given properties change. Lowering opacity is a good way to tone down a "
            "whole change at once.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "changes": {
                        "type": "object",
                        "description": "Layer properties to values: " + ", ".join(layer_props),
                    },
                },
                "required": ["id", "changes"],
                "additionalProperties": False,
            },
            "eager_input_streaming": True,
        }
    )
    tools.append(
        {
            "name": "remove_layer",
            "description": "Remove a layer and all its operations, by id.",
            "input_schema": {
                "type": "object",
                "properties": {"id": {"type": "string"}},
                "required": ["id"],
                "additionalProperties": False,
            },
            "eager_input_streaming": True,
        }
    )
    cutout_schema = Cutout.model_json_schema()
    cutout_props = {k: v for k, v in cutout_schema["properties"].items() if k != "visible"}
    cut_out_schema: dict[str, Any] = {
        "type": "object",
        "properties": cutout_props,
        "additionalProperties": False,
    }
    if "$defs" in cutout_schema:
        cut_out_schema["$defs"] = _referenced_defs(cutout_props, cutout_schema["$defs"])
    tools.append(
        {
            "name": "cut_out",
            "description": "Remove the background: keep only what the mask selects (by "
            "default the main subject, found by an AI model) and make the rest transparent, "
            "or a solid color. Replaces any earlier cutout.",
            "input_schema": _plain_unions(cut_out_schema),
            "eager_input_streaming": True,
        }
    )
    retouch_schema = Retouch.model_json_schema()
    tools.append(
        {
            "name": "retouch_portrait",
            "description": "Retouch the people in a portrait: heal blemishes, smooth skin "
            "while keeping its texture, brighten eyes, and whiten teeth. Adds one layer per "
            "part, each masked to that part of the face, so each can be tuned or hidden. "
            "The defaults are deliberately subtle; set a part to 0 to leave it out.",
            "input_schema": {
                "type": "object",
                "properties": retouch_schema["properties"],
                "additionalProperties": False,
            },
            "eager_input_streaming": True,
        }
    )
    tools.append(
        {
            "name": "auto_straighten",
            "description": "Level the photo and square up converging verticals, measured "
            "from the straight lines in it (horizon, walls, buildings). Replaces any "
            "straighten or perspective already in the framing; keeps crops, rotations, and "
            "flips. Reports what it found, or that the photo gave no clear lines.",
            "input_schema": {
                "type": "object",
                "properties": AutoStraighten.model_json_schema()["properties"],
                "additionalProperties": False,
            },
            "eager_input_streaming": True,
        }
    )
    tools.append(
        {
            "name": "restore_background",
            "description": "Undo the cutout and bring the background back.",
            "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
            "eager_input_streaming": True,
        }
    )
    tools.append(
        {
            "name": "update_operation",
            "description": "Change parameters of an operation already present (in the framing "
            "or any layer), by id. Only the given parameters change.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "changes": {"type": "object", "description": "Parameter names to values."},
                },
                "required": ["id", "changes"],
                "additionalProperties": False,
            },
            "eager_input_streaming": True,
        }
    )
    tools.append(
        {
            "name": "remove_operation",
            "description": "Remove an operation (from the framing or any layer), by id.",
            "input_schema": {
                "type": "object",
                "properties": {"id": {"type": "string"}},
                "required": ["id"],
                "additionalProperties": False,
            },
            "eager_input_streaming": True,
        }
    )
    return tools


TOOLS = tool_definitions()


class ToolError(Exception):
    pass


class Editor:
    """Applies tool calls to a working copy of the edit state."""

    def __init__(
        self,
        state: EditState,
        default_layer_name: str = "Edits",
        framed: Framed | None = None,
        model_for: Callable[[str], str] | None = None,
    ) -> None:
        self.state = state.model_copy(deep=True)
        self.default_layer_name = default_layer_name
        self.framed = framed
        """Renders the photo with a given framing, for tools that measure it."""
        self.model_for = model_for
        """Names the backend (model or "classical") that runs a model task."""
        self.target: str | None = None
        """The layer that operation tools add to: the one added most recently."""

    def call(self, name: str, args: object) -> tuple[str, OperationEvent]:
        result = self._call(name, args)
        # Generative edits get their seed and model now, so every render shows the same take.
        self.state = stamp(self.state, self.model_for)
        return result

    def _call(self, name: str, args: object) -> tuple[str, OperationEvent]:
        if not isinstance(args, dict):
            raise ToolError("Tool input must be a JSON object.")
        handler = {
            "update_operation": self._update,
            "remove_operation": self._remove,
            "add_layer": self._add_layer,
            "update_layer": self._update_layer,
            "remove_layer": self._remove_layer,
            "cut_out": self._cut_out,
            "retouch_portrait": self._retouch,
            "auto_straighten": self._auto_straighten,
            "restore_background": self._restore_background,
        }.get(name)
        if handler is not None:
            return handler(args)
        if name not in OPERATIONS_BY_NAME:
            raise ToolError(f"Unknown tool {name!r}.")
        op = self._validate({**args, "op": name})
        if isinstance(op, GEOMETRY_TYPES):
            self.state.framing.append(op)  # type: ignore[arg-type]
            where = "the framing"
        else:
            layer = self._target_layer()
            if isinstance(op, CONTENT_TYPES) and not isinstance(op, MASKED_TYPES):
                if layer.operations:
                    # It works without a mask, so it can start its own layer.
                    layer = self._new_layer(op.summary())
            elif isinstance(op, CONTENT_TYPES) and (layer.operations or layer.mask is None):
                raise ToolError(
                    f"{name} needs its own new layer whose mask selects where it applies: "
                    "call add_layer with that mask first."
                )
            if layer.is_content:
                raise ToolError(
                    f"Layer {layer.id} changes what is in the photo; add a new layer for "
                    "adjustments."
                )
            layer.operations.append(op)  # type: ignore[arg-type]
            where = f"layer {layer.id} ({layer.name})"
        summary = op.summary()
        text = f"Added {summary} as operation {op.id} in {where}."
        if isinstance(op, GenerativeBase):
            text += self._generative_note(op)
        return text, OperationEvent(action="added", summary=summary)

    def _generative_note(self, op: GenerativeBase) -> str:
        if self.model_for is None or self.model_for(task_for(op)) != "classical":
            return ""
        return (
            " Note: no generative model is installed here, so this is only a rough stand-in "
            "(for a fill, the surroundings continued into it). Tell the person, and that "
            "installing the models (uv sync --extra models) gives the real result."
        )

    def _target_layer(self) -> Layer:
        if self.target is not None:
            try:
                return self.state.layer(self.target)
            except KeyError:
                pass
        return self._new_layer(self.default_layer_name)

    def _new_layer(self, name: str) -> Layer:
        layer = Layer(id=new_layer_id(), name=name[:80])
        self.state.layers.append(layer)
        self.target = layer.id
        return layer

    def _containers(self) -> list[tuple[list[Any], Layer | None]]:
        return [(self.state.framing, None)] + [(lay.operations, lay) for lay in self.state.layers]

    def _find(self, op_id: object) -> tuple[list[Any], int]:
        for ops, _ in self._containers():
            for i, op in enumerate(ops):
                if op.id == op_id:
                    return ops, i
        raise ToolError(f"No operation with id {op_id!r}.")

    def _find_layer(self, layer_id: object) -> Layer:
        for layer in self.state.layers:
            if layer.id == layer_id:
                return layer
        raise ToolError(f"No layer with id {layer_id!r}.")

    def _update(self, args: dict[str, Any]) -> tuple[str, OperationEvent]:
        ops, index = self._find(args.get("id"))
        changes = args.get("changes")
        if not isinstance(changes, dict):
            raise ToolError("changes must be an object of parameter names to values.")
        current = ops[index]
        changes = {k: v for k, v in changes.items() if k not in ("id", "op")}
        op = self._validate({**current.model_dump(), **changes})
        ops[index] = op
        summary = op.summary()
        return f"Updated operation {op.id}: {summary}.", OperationEvent(
            action="updated", summary=summary
        )

    def _remove(self, args: dict[str, Any]) -> tuple[str, OperationEvent]:
        ops, index = self._find(args.get("id"))
        op = ops.pop(index)
        summary = op.summary()
        return f"Removed operation {op.id} ({summary}).", OperationEvent(
            action="removed", summary=summary
        )

    def _add_layer(self, args: dict[str, Any]) -> tuple[str, OperationEvent]:
        layer = self._validate_layer({**args, "id": new_layer_id()})
        self.state.layers.append(layer)
        self.target = layer.id
        return (
            f"Added layer {layer.id} ({layer.name}) on top. Operation tools now add to this layer.",
            OperationEvent(action="added", summary=f"New layer: {layer.name}"),
        )

    def _update_layer(self, args: dict[str, Any]) -> tuple[str, OperationEvent]:
        layer = self._find_layer(args.get("id"))
        changes = args.get("changes")
        if not isinstance(changes, dict):
            raise ToolError("changes must be an object of layer properties to values.")
        changes = {k: v for k, v in changes.items() if k not in ("id", "operations")}
        mask = changes.get("mask")
        old = layer.mask
        if (
            isinstance(mask, dict)
            and "strokes" not in mask
            and isinstance(old, SemanticMask)
            and old.strokes
            and mask.get("kind") == "semantic"
            and mask.get("target") == old.target
        ):
            # Keep the person's brush touch-ups on a selection the agent only adjusts.
            changes["mask"] = {**mask, "strokes": [s.model_dump() for s in old.strokes]}
        updated = self._validate_layer({**layer.model_dump(), **changes})
        index = self.state.layers.index(layer)
        self.state.layers[index] = updated
        return f"Updated layer {layer.id}.", OperationEvent(
            action="updated", summary=f"Layer: {updated.name} ({describe_layer(updated)})"
        )

    def _cut_out(self, args: dict[str, Any]) -> tuple[str, OperationEvent]:
        try:
            cutout = Cutout.model_validate({k: v for k, v in args.items() if k != "visible"})
        except ValidationError as exc:
            raise ToolError(f"Invalid cutout: {_problems(exc)}") from None
        self.state.cutout = cutout
        background = cutout.background or "transparent"
        return f"Cut out with a {background} background.", OperationEvent(
            action="added", summary=f"Background removed ({background})"
        )

    def _retouch(self, args: dict[str, Any]) -> tuple[str, OperationEvent]:
        try:
            options = Retouch.model_validate(args)
        except ValidationError as exc:
            raise ToolError(f"Invalid retouch: {_problems(exc)}") from None
        layers = retouch_layers(options)
        if not layers:
            raise ToolError("Every part of the retouch was set to 0.")
        self.state.layers.extend(layers)
        names = ", ".join(f"{layer.id} ({layer.name})" for layer in layers)
        return f"Added retouch layers {names}.", OperationEvent(
            action="added", summary="Portrait retouch: " + ", ".join(lay.name for lay in layers)
        )

    def _auto_straighten(self, args: dict[str, Any]) -> tuple[str, OperationEvent]:
        try:
            options = AutoStraighten.model_validate(args)
        except ValidationError as exc:
            raise ToolError(f"Invalid auto_straighten: {_problems(exc)}") from None
        if self.framed is None:
            raise ToolError("The photo is not available to measure.")
        found = auto_level(self.state.framing, self.framed, options)
        done = found.describe()
        if not done:
            raise ToolError(
                "Found no clear horizon or verticals to go by, or the photo is already "
                "straight; nothing changed. Set straighten yourself if it still looks tilted."
            )
        self.state.framing = found.framing  # type: ignore[assignment]
        return f"Added {done} to the framing.", OperationEvent(
            action="added", summary=f"Auto straighten ({done})"
        )

    def _restore_background(self, args: dict[str, Any]) -> tuple[str, OperationEvent]:
        if self.state.cutout is None:
            raise ToolError("There is no cutout to undo.")
        self.state.cutout = None
        return "Restored the background.", OperationEvent(
            action="removed", summary="Background removal"
        )

    def _remove_layer(self, args: dict[str, Any]) -> tuple[str, OperationEvent]:
        layer = self._find_layer(args.get("id"))
        self.state.layers.remove(layer)
        return f"Removed layer {layer.id} ({layer.name}).", OperationEvent(
            action="removed", summary=f"Layer: {layer.name}"
        )

    @staticmethod
    def _validate(data: dict[str, Any]) -> Operation:
        try:
            return OperationAdapter.validate_python(data)
        except ValidationError as exc:
            raise ToolError(f"Invalid parameters: {_problems(exc)}") from None

    @staticmethod
    def _validate_layer(data: dict[str, Any]) -> Layer:
        try:
            return Layer.model_validate(data)
        except ValidationError as exc:
            raise ToolError(f"Invalid layer: {_problems(exc)}") from None


def layer_name(request: str) -> str:
    """A layer name for edits made without add_layer: the start of the request."""
    words = " ".join(request.split())
    name = words if len(words) <= 40 else words[:39].rsplit(" ", 1)[0] + "…"
    return (name[:1].upper() + name[1:]) or "Edits"


def _problems(exc: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(p) for p in e['loc']) or 'input'}: {e['msg']}" for e in exc.errors()
    )


def describe_layer(layer: Layer) -> str:
    parts = [] if layer.visible else ["hidden"]
    parts.append(f"opacity {layer.opacity:g}%")
    if layer.blend_mode != "normal":
        parts.append(f"blend {layer.blend_mode}")
    if layer.mask is not None:
        parts.append(describe_mask(layer.mask))
    return ", ".join(parts)


def _op_json(op: OpBase) -> str:
    return json.dumps(op.model_dump(), separators=(",", ":"))


def describe_state(state: EditState) -> str:
    if state.is_empty:
        return "No edits yet; this is the original photo."
    lines = ["Current edits."]
    if state.framing:
        lines.append("Framing, applied first:")
        lines += [f"  {_op_json(op)}" for op in state.framing]
    else:
        lines.append("Framing: none.")
    if state.layers:
        lines.append("Layers, bottom to top:")
        for layer in state.layers:
            lines.append(f'- layer {layer.id} "{layer.name}" ({describe_layer(layer)})')
            lines += [f"    {_op_json(op)}" for op in layer.operations] or ["    (empty)"]
    else:
        lines.append("Layers: none.")
    cutout = state.cutout
    if cutout is not None:
        hidden = "" if cutout.visible else " (hidden)"
        background = cutout.background or "transparent"
        lines.append(
            f"Cutout{hidden}, applied last: keeps the {describe_mask(cutout.mask)}, "
            f"{background} background."
        )
    return "\n".join(lines)


def image_block(pixels: imaging.Array) -> BetaImageBlockParam:
    small = imaging.resize_long_edge(pixels, CLAUDE_IMAGE_LONG_EDGE)
    data = base64.standard_b64encode(imaging.encode_jpeg(small, quality=85)).decode()
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}}


def events_note(events: Sequence[str]) -> str:
    if not events:
        return ""
    return f"(Since your last reply, the person used: {'; '.join(events)}.)\n\n"


def history_messages(chat: Sequence[ChatEntry]) -> tuple[list[BetaMessageParam], list[str]]:
    """Replay the conversation so far as alternating user and assistant text.

    Also returns history events (undo, redo, jumps) that happened after the last reply, for
    the next request.
    """
    messages: list[BetaMessageParam] = []
    pending_events: list[str] = []
    for entry in chat:
        if entry.role == "event":
            pending_events.append(entry.text)
            continue
        text = entry.text
        if entry.role == "user" and pending_events:
            text = events_note(pending_events) + text
            pending_events = []
        if messages and messages[-1]["role"] == entry.role:
            prev = messages[-1]["content"]
            assert isinstance(prev, str)
            messages[-1] = {"role": entry.role, "content": f"{prev}\n\n{text}"}
        else:
            messages.append({"role": entry.role, "content": text})
    if messages and messages[0]["role"] == "assistant":
        messages.insert(0, {"role": "user", "content": "(Photo opened.)"})
    if messages and messages[-1]["role"] == "user":
        # An earlier turn failed before replying; keep roles alternating.
        messages.append({"role": "assistant", "content": "(No reply.)"})
    return messages, pending_events


class AgentService:
    def __init__(self, store: DocumentStore, renders: RenderCache, model: ModelClient) -> None:
        self.store = store
        self.renders = renders
        self.model = model
        self._locks: dict[str, asyncio.Lock] = {}

    async def run_turn(self, doc_id: str, request: str, emit: Emit) -> Document:
        """Handle one chat message: let Claude edit, then record the turn and its reply."""
        lock = self._locks.setdefault(doc_id, asyncio.Lock())
        async with lock:
            doc = self.store.get(doc_id)
            await emit(TurnStarted())
            loaded = self.store.image(doc_id)
            editor = Editor(
                doc.state,
                default_layer_name=layer_name(request),
                framed=lambda framing: render(loaded.proxy, framing, loaded.proxy_context),
                model_for=self.store.model_for,
            )
            reply = await self._converse(doc, request, editor, emit)

            doc.chat.append(ChatEntry(role="user", text=request))
            step_id = None
            if editor.state != doc.state:
                step = Step(
                    kind="agent",
                    label=request,
                    request=request,
                    reply=reply,
                    state=editor.state,
                )
                doc.commit(step)
                step_id = step.id
            doc.chat.append(ChatEntry(role="assistant", text=reply, step_id=step_id))
            self.store.save(doc)
            await emit(TurnDone(document=DocumentView.of(doc)))
            return doc

    async def _render(self, doc: Document, state: EditState) -> imaging.Array:
        loaded = self.store.image(doc.id)
        return await asyncio.to_thread(
            self.renders.get_or_render, doc.id, loaded.proxy, state, loaded.proxy_context
        )

    async def _attach_self_check(
        self,
        doc: Document,
        editor: Editor,
        results: list[dict[str, Any]],
        baseline: diagnostics.Measurements,
    ) -> None:
        """Show Claude what its edits did, so it can catch overshoots before replying."""
        rendered = await self._render(doc, editor.state)
        # Ops changed, so at least one call succeeded; annotate the last successful one.
        last = next(r for r in reversed(results) if not r.get("is_error"))
        last["content"] = [
            {"type": "text", "text": str(last["content"])},
            {"type": "text", "text": diagnostics.report(baseline, diagnostics.measure(rendered))},
            {"type": "text", "text": describe_state(editor.state)},
            image_block(rendered),
        ]

    async def _converse(self, doc: Document, request: str, editor: Editor, emit: Emit) -> str:
        history, events = history_messages(doc.chat)
        preview = await self._render(doc, editor.state)
        baseline = diagnostics.measure(await self._render(doc, EditState()))
        state_seen = editor.state.model_copy(deep=True)
        intro = events_note(events)
        messages: list[BetaMessageParam] = [
            *history,
            {
                "role": "user",
                "content": [
                    image_block(preview),
                    {
                        "type": "text",
                        "text": f"{describe_state(editor.state)}\n\n{intro}Request: {request}",
                    },
                ],
            },
        ]
        reply_parts: list[str] = []
        new_paragraph = False

        async def on_text(delta: str) -> None:
            nonlocal new_paragraph
            if new_paragraph and reply_parts:
                delta = "\n\n" + delta.lstrip()
            new_paragraph = False
            reply_parts.append(delta)
            await emit(TextDelta(text=delta))

        for _ in range(MAX_MODEL_CALLS):
            # Text from separate model calls reads as separate paragraphs.
            new_paragraph = True
            response = await self.model.create(
                system=SYSTEM_PROMPT, tools=TOOLS, messages=messages, on_text=on_text
            )
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "refusal":
                await on_text("Sorry, I can't help with that request.")
                break
            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if not tool_uses:
                break
            if response.stop_reason == "max_tokens":
                raise RuntimeError("The agent's reply was cut off mid-edit.")

            results: list[dict[str, Any]] = []
            for block in tool_uses:
                try:
                    text, event = editor.call(block.name, block.input)
                except ToolError as exc:
                    results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": str(exc),
                            "is_error": True,
                        }
                    )
                    continue
                await emit(event)
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": text})
            if editor.state != state_seen:
                state_seen = editor.state.model_copy(deep=True)
                await self._attach_self_check(doc, editor, results, baseline)
            messages.append({"role": "user", "content": results})  # type: ignore[typeddict-item]
        else:
            new_paragraph = True
            await on_text("I stopped here to keep things quick; tell me if you want more changes.")

        if not "".join(reply_parts).strip():
            await on_text("Done." if editor.state != doc.state else "OK.")
        return "".join(reply_parts).strip()
