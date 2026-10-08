"""A scripted stand-in for Claude, so agent tests run offline and deterministically."""

from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from anthropic.types.beta import (
    BetaMessage,
    BetaMessageParam,
    BetaTextBlock,
    BetaToolParam,
    BetaToolUseBlock,
)


def text(t: str) -> BetaTextBlock:
    return BetaTextBlock.model_construct(type="text", text=t, citations=None)


def tool(name: str, args: dict[str, Any], id: str | None = None) -> BetaToolUseBlock:
    return BetaToolUseBlock.model_construct(
        type="tool_use", id=id or f"toolu_{name}_{len(args)}", name=name, input=args
    )


class FakeModel:
    """Replies with the scripted responses in order and records every request it saw."""

    def __init__(
        self,
        *responses: list[Any] | Callable[[list[dict[str, Any]]], list[Any]],
        stop_reason: str | None = None,
    ) -> None:
        self.responses = list(responses)
        self.stop_reason = stop_reason
        self.calls: list[list[dict[str, Any]]] = []
        """Each request's messages, loosely typed so tests can poke at them. A scripted
        response may be a function of the request's messages."""

    async def create(
        self,
        *,
        system: str,
        tools: Sequence[BetaToolParam],
        messages: Sequence[BetaMessageParam],
        on_text: Callable[[str], Awaitable[None]],
    ) -> BetaMessage:
        self.calls.append([dict(m) for m in messages])
        scripted = self.responses.pop(0)
        blocks = scripted(self.calls[-1]) if callable(scripted) else scripted
        for block in blocks:
            if block.type == "text":
                await on_text(block.text)
        has_tools = any(b.type == "tool_use" for b in blocks)
        fields: dict[str, Any] = {
            "id": "msg_fake",
            "type": "message",
            "role": "assistant",
            "model": "fake",
            "content": blocks,
            "stop_reason": self.stop_reason or ("tool_use" if has_tools else "end_turn"),
            "stop_sequence": None,
        }
        return BetaMessage.model_construct(**fields)
