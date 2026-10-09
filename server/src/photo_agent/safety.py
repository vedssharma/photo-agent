"""Safety screen for generative edits.

Generative edits can make a photo show something that never happened, so each new prompt
is checked before anything is generated. Two layers:

- Local rules, always on: sexual content involving minors, undressing or sexualizing real
  people in a photo, and the most obvious deceptive requests.
- Claude, when an API key is configured: a short classification of the request, for what
  rules cannot catch, chiefly deceptive impersonation of real people (putting a public
  figure, or anyone, somewhere they never were in a way meant to be passed off as real).

Ordinary creative and personal edits (a new sky, a beach background, a fantasy scene, a
caricature that is plainly art) are fine. If Claude cannot be reached, the local rules
still apply and the edit goes ahead; the check is a guard against misuse, not a gate on
every edit. Verdicts are cached, since the same prompt is often re-rendered.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from collections import OrderedDict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from anthropic import Anthropic

from photo_agent.layers import EditState
from photo_agent.operations import GenerativeBase

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Verdict:
    allowed: bool
    reason: str = ""


ALLOWED = Verdict(True)

MINORS = (
    r"\b(child|children|kid|kids|minor|minors|teen|teens|teenage|underage|boy|girl|baby"
    r"|toddler|schoolgirl|schoolboy)\b"
)
SEXUAL = (
    r"\b(nude|naked|nudity|topless|undress(ed|ing)?|lingerie|sexy|sexual|porn\w*|nsfw"
    r"|erotic)\b"
)
UNDRESS = (
    r"\b(remove|take off|without|no)\b.{0,20}"
    r"\b(clothes|clothing|shirt|bra|underwear|bikini|swimsuit)\b"
)
DECEPTIVE = (
    r"\b(fake (id|passport|license|document|receipt|evidence|crime scene)|forged?"
    r"|counterfeit)\b"
)

RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(SEXUAL, re.I), "I can't make sexual or nude edits of photos."),
    (re.compile(UNDRESS, re.I), "I can't remove or change people's clothing to reveal them."),
    (re.compile(DECEPTIVE, re.I), "I can't help fake documents or evidence."),
]


def local_rules(prompt: str) -> Verdict:
    """The rules that apply without a model."""
    if re.search(MINORS, prompt, re.I) and re.search(SEXUAL, prompt, re.I):
        return Verdict(False, "I can't make that edit.")
    for pattern, reason in RULES:
        if pattern.search(prompt):
            return Verdict(False, reason)
    return ALLOWED


SYSTEM = """You review requests to a consumer photo editor before it runs a generative \
image model on the person's own photo. Decide whether the edit is allowed.

Allowed (the large majority): creative and personal edits, including adding or removing \
objects, new skies and backgrounds, putting the person somewhere new, fantasy and art \
styles, relighting, restoring old photos, and edits involving public places or landmarks.

Blocked:
- Deceptive impersonation: adding a real, identifiable person (a celebrity, politician, \
or any named individual) to the photo, or making someone appear to do or endorse something \
they did not, when the result could pass as a real photo. Plain caricature or obvious art \
is fine.
- Sexual or nude content of any real person, and anything sexual involving minors.
- Faked evidence or documents (IDs, receipts, injuries for an insurance claim, crime \
scenes), and violent or hateful content aimed at a real person.

If a request is ambiguous, allow it. Give a short reason, written to the person, only \
when blocking."""

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "allowed": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["allowed", "reason"],
    "additionalProperties": False,
}

Classifier = Callable[[str, str], Verdict]


class ClaudeClassifier:
    """Asks Claude whether a generative request is allowed."""

    def __init__(self, api_key: str, model: str) -> None:
        self.client = Anthropic(api_key=api_key, max_retries=2, timeout=30)
        self.model = model

    def __call__(self, op: str, prompt: str) -> Verdict:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=SYSTEM,
            output_config={"format": {"type": "json_schema", "schema": SCHEMA}, "effort": "low"},
            messages=[
                {
                    "role": "user",
                    "content": f"Edit: {op.replace('_', ' ')}\nRequest: {prompt}",
                }
            ],
        )
        if response.stop_reason == "refusal":
            return Verdict(False, "I can't make that edit.")
        text = next(block.text for block in response.content if block.type == "text")
        answer = json.loads(text)
        if answer["allowed"]:
            return ALLOWED
        return Verdict(False, str(answer["reason"]) or "I can't make that edit.")


class Screen:
    """Checks the prompts of generative edits, remembering verdicts."""

    MEMORY = 512

    def __init__(self, classifier: Classifier | None = None) -> None:
        self.classifier = classifier
        self._verdicts: OrderedDict[tuple[str, str], Verdict] = OrderedDict()
        self._lock = threading.Lock()

    def check(self, op: str, prompt: str) -> Verdict:
        prompt = prompt.strip()
        if not prompt:
            return ALLOWED
        key = (op, prompt)
        with self._lock:
            if key in self._verdicts:
                return self._verdicts[key]
        verdict = local_rules(prompt)
        if verdict.allowed and self.classifier is not None:
            try:
                verdict = self.classifier(op, prompt)
            except Exception as exc:  # the API is down or answered oddly
                log.warning("Safety check unavailable, using local rules only: %s", exc)
                return verdict  # not remembered, so it is asked again next time
        with self._lock:
            self._verdicts[key] = verdict
            while len(self._verdicts) > self.MEMORY:
                self._verdicts.popitem(last=False)
        return verdict

    def check_new(self, before: EditState, after: EditState) -> Verdict:
        """The first blocking verdict among prompts in `after` that `before` lacks."""
        seen = set(_prompts(before.all_operations()))
        for key in _prompts(after.all_operations()):
            if key not in seen:
                verdict = self.check(*key)
                if not verdict.allowed:
                    return verdict
        return ALLOWED


def _prompts(operations: Iterable[object]) -> Iterable[tuple[str, str]]:
    for op in operations:
        if isinstance(op, GenerativeBase):
            prompt = getattr(op, "prompt", "")
            if prompt:
                yield op.op, prompt  # type: ignore[attr-defined]
