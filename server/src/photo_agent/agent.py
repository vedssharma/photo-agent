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
from photo_agent.operations import OPERATIONS_BY_NAME, OpBase, Operation, OperationAdapter
from photo_agent.render import RenderCache
from photo_agent.store import DocumentStore

MAX_MODEL_CALLS = 8
"""Upper bound on model round trips in one turn, so a confused agent cannot loop forever."""
CLAUDE_IMAGE_LONG_EDGE = 1024

SYSTEM_PROMPT = """\
You are the photo editor inside photo-agent. The people you help are not editing experts; \
they describe what they want in everyday words and expect a professional-looking result.

How editing works:
- You never change pixels yourself. You edit by calling tools. Each tool call adds one \
operation to the photo's ordered operation list, which is rendered in order on top of the \
untouched original. update_operation and remove_operation change operations already in the \
list; prefer adjusting an existing operation over stacking a second one of the same kind.
- With each request you get the current rendered photo and the current operation list.
- Crop boxes are fractions of the frame as it is at that point in the list, after any \
earlier rotate, straighten, or crop. For Instagram, use 4:5 for portrait feed posts, 1:1 for \
square, 1.91:1 is not available so use 16:9 for landscape, and 9:16 for stories.

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
    tools.append(
        {
            "name": "update_operation",
            "description": "Change parameters of an operation already in the list, by id. "
            "Only the given parameters change.",
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
            "description": "Remove an operation from the list, by id.",
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
    """Applies tool calls to a working copy of the operation list."""

    def __init__(self, operations: Sequence[Operation]) -> None:
        self.operations: list[Operation] = list(operations)

    def call(self, name: str, args: object) -> tuple[str, OperationEvent]:
        if not isinstance(args, dict):
            raise ToolError("Tool input must be a JSON object.")
        if name == "update_operation":
            return self._update(args)
        if name == "remove_operation":
            return self._remove(args)
        if name not in OPERATIONS_BY_NAME:
            raise ToolError(f"Unknown tool {name!r}.")
        op = self._validate({**args, "op": name})
        self.operations.append(op)
        summary = op.summary()
        return f"Added {summary} as operation {op.id}.", OperationEvent(
            action="added", summary=summary
        )

    def _find(self, op_id: object) -> int:
        for i, op in enumerate(self.operations):
            if op.id == op_id:
                return i
        raise ToolError(f"No operation with id {op_id!r}.")

    def _update(self, args: dict[str, Any]) -> tuple[str, OperationEvent]:
        index = self._find(args.get("id"))
        changes = args.get("changes")
        if not isinstance(changes, dict):
            raise ToolError("changes must be an object of parameter names to values.")
        current = self.operations[index]
        changes = {k: v for k, v in changes.items() if k not in ("id", "op")}
        op = self._validate({**current.model_dump(), **changes})
        self.operations[index] = op
        summary = op.summary()
        return f"Updated operation {op.id}: {summary}.", OperationEvent(
            action="updated", summary=summary
        )

    def _remove(self, args: dict[str, Any]) -> tuple[str, OperationEvent]:
        index = self._find(args.get("id"))
        op = self.operations.pop(index)
        summary = op.summary()
        return f"Removed operation {op.id} ({summary}).", OperationEvent(
            action="removed", summary=summary
        )

    @staticmethod
    def _validate(data: dict[str, Any]) -> Operation:
        try:
            return OperationAdapter.validate_python(data)
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(str(p) for p in e['loc']) or 'input'}: {e['msg']}" for e in exc.errors()
            )
            raise ToolError(f"Invalid parameters: {problems}") from None


def describe_operations(operations: Sequence[OpBase]) -> str:
    if not operations:
        return "No edits yet; this is the original photo."
    lines = [json.dumps(op.model_dump(), separators=(",", ":")) for op in operations]
    return "Current operation list, in render order:\n" + "\n".join(lines)


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
            editor = Editor(doc.operations)
            reply = await self._converse(doc, request, editor, emit)

            doc.chat.append(ChatEntry(role="user", text=request))
            step_id = None
            if editor.operations != doc.operations:
                step = Step(
                    kind="agent",
                    label=request,
                    request=request,
                    reply=reply,
                    operations=editor.operations,
                )
                doc.commit(step)
                step_id = step.id
            doc.chat.append(ChatEntry(role="assistant", text=reply, step_id=step_id))
            self.store.save(doc)
            await emit(TurnDone(document=DocumentView.of(doc)))
            return doc

    async def _render(self, doc: Document, operations: Sequence[Operation]) -> imaging.Array:
        loaded = self.store.image(doc.id)
        return await asyncio.to_thread(
            self.renders.get_or_render, doc.id, loaded.proxy, operations, loaded.proxy_context
        )

    async def _attach_self_check(
        self,
        doc: Document,
        editor: Editor,
        results: list[dict[str, Any]],
        baseline: diagnostics.Measurements,
    ) -> None:
        """Show Claude what its edits did, so it can catch overshoots before replying."""
        rendered = await self._render(doc, editor.operations)
        # Ops changed, so at least one call succeeded; annotate the last successful one.
        last = next(r for r in reversed(results) if not r.get("is_error"))
        last["content"] = [
            {"type": "text", "text": str(last["content"])},
            {"type": "text", "text": diagnostics.report(baseline, diagnostics.measure(rendered))},
            {"type": "text", "text": describe_operations(editor.operations)},
            image_block(rendered),
        ]

    async def _converse(self, doc: Document, request: str, editor: Editor, emit: Emit) -> str:
        history, events = history_messages(doc.chat)
        preview = await self._render(doc, editor.operations)
        baseline = diagnostics.measure(await self._render(doc, []))
        operations_seen = list(editor.operations)
        intro = events_note(events)
        messages: list[BetaMessageParam] = [
            *history,
            {
                "role": "user",
                "content": [
                    image_block(preview),
                    {
                        "type": "text",
                        "text": f"{describe_operations(editor.operations)}\n\n"
                        f"{intro}Request: {request}",
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
            if editor.operations != operations_seen:
                operations_seen = list(editor.operations)
                await self._attach_self_check(doc, editor, results, baseline)
            messages.append({"role": "user", "content": results})  # type: ignore[typeddict-item]
        else:
            new_paragraph = True
            await on_text("I stopped here to keep things quick; tell me if you want more changes.")

        if not "".join(reply_parts).strip():
            await on_text("Done." if editor.operations != doc.operations else "OK.")
        return "".join(reply_parts).strip()
