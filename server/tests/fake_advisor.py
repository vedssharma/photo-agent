"""A scripted stand-in for one-shot structured questions to Claude."""

from collections.abc import Sequence
from typing import Any

from photo_agent.advisor import Tier


class FakeAdvisor:
    """Answers with the scripted answers in order and records what it was asked."""

    def __init__(self, *answers: dict[str, Any] | Exception) -> None:
        self.answers = list(answers)
        self.calls: list[dict[str, Any]] = []

    async def ask(
        self,
        *,
        system: str,
        content: Sequence[Any],
        schema: dict[str, Any],
        tier: Tier = "deep",
    ) -> dict[str, Any]:
        self.calls.append({"system": system, "content": list(content), "tier": tier})
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer
