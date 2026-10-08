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
from photo_agent.graph import ChatEntry, Document, DocumentView, Step
from photo_agent.layers import BLEND_MODE_HELP, EditState, Layer, new_layer_id
from photo_agent.operations import (
    GEOMETRY_TYPES,
    OPERATIONS_BY_NAME,
    OpBase,
    Operation,
    OperationAdapter,
)
from photo_agent.render import RenderCache
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
- If a request needs something the tools cannot do (removing objects, changing the sky, \
generating content), say so plainly and offer what you can do instead.
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


def tool_definitions() -> list[BetaToolParam]:
    tools: list[BetaToolParam] = []
    for name, cls in OPERATIONS_BY_NAME.items():
        schema = cls.model_json_schema()
        props = {k: v for k, v in schema["properties"].items() if k not in ("id", "op")}
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
        add_layer_schema["$defs"] = layer_schema["$defs"]
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
            "description": "Change a layer's name, visibility, opacity, or blend mode, by id. "
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

    def __init__(self, state: EditState, default_layer_name: str = "Edits") -> None:
        self.state = state.model_copy(deep=True)
        self.default_layer_name = default_layer_name
        self.target: str | None = None
        """The layer that operation tools add to: the one added most recently."""

    def call(self, name: str, args: object) -> tuple[str, OperationEvent]:
        if not isinstance(args, dict):
            raise ToolError("Tool input must be a JSON object.")
        handler = {
            "update_operation": self._update,
            "remove_operation": self._remove,
            "add_layer": self._add_layer,
            "update_layer": self._update_layer,
            "remove_layer": self._remove_layer,
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
            layer.operations.append(op)  # type: ignore[arg-type]
            where = f"layer {layer.id} ({layer.name})"
        summary = op.summary()
        return f"Added {summary} as operation {op.id} in {where}.", OperationEvent(
            action="added", summary=summary
        )

    def _target_layer(self) -> Layer:
        if self.target is not None:
            try:
                return self.state.layer(self.target)
            except KeyError:
                pass
        layer = Layer(id=new_layer_id(), name=self.default_layer_name)
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
        updated = self._validate_layer({**layer.model_dump(), **changes})
        index = self.state.layers.index(layer)
        self.state.layers[index] = updated
        return f"Updated layer {layer.id}.", OperationEvent(
            action="updated", summary=f"Layer: {updated.name} ({describe_layer(updated)})"
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
            editor = Editor(doc.state, default_layer_name=layer_name(request))
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
