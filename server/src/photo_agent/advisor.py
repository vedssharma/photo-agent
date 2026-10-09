"""One-shot questions to Claude that come back as structured data: suggestions, critiques.

The chat agent edits through a tool loop (`photo_agent.agent`). Some features only need
Claude's judgment about a photo, returned in a fixed shape: "what directions could this
photo go in?", "what is working and what is not?". Those ask once, with a JSON schema for
the answer (structured outputs), and never edit anything themselves.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, Literal, Protocol

from anthropic import AsyncAnthropic
from anthropic.types.beta import BetaContentBlockParam

Tier = Literal["routine", "deep"]
"""How much judgment a call needs: routine calls can go to a smaller, faster model."""


class AdvisorError(RuntimeError):
    """Claude could not give an answer (a refusal, or one that was cut off)."""


class Advisor(Protocol):
    """Asks Claude one question about a photo. Implemented by `ClaudeAdvisor`, and by fakes
    in tests."""

    async def ask(
        self,
        *,
        system: str,
        content: Sequence[BetaContentBlockParam],
        schema: dict[str, Any],
        tier: Tier = "deep",
    ) -> dict[str, Any]: ...


class ClaudeAdvisor:
    def __init__(self, api_key: str, model: str) -> None:
        self.client = AsyncAnthropic(api_key=api_key, max_retries=2, timeout=120)
        self.model = model

    def model_for(self, tier: Tier) -> str:
        return self.model

    async def ask(
        self,
        *,
        system: str,
        content: Sequence[BetaContentBlockParam],
        schema: dict[str, Any],
        tier: Tier = "deep",
    ) -> dict[str, Any]:
        response = await self.client.beta.messages.create(
            model=self.model_for(tier),
            max_tokens=16000,
            system=system,
            messages=[{"role": "user", "content": list(content)}],
            thinking={"type": "adaptive"},
            output_config={
                "format": {"type": "json_schema", "schema": schema},
                "effort": "medium" if tier == "deep" else "low",
            },
        )
        if response.stop_reason == "refusal":
            raise AdvisorError("Claude declined to look at this photo.")
        if response.stop_reason == "max_tokens":
            raise AdvisorError("Claude's answer was cut off.")
        text = "".join(block.text for block in response.content if block.type == "text")
        answer = json.loads(text)
        if not isinstance(answer, dict):
            raise AdvisorError("Claude answered in an unexpected shape.")
        return answer
