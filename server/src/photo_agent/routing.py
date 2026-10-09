"""Cost and latency controls: which Claude model handles a turn.

Most chat turns are routine ("a bit warmer", "less contrast", "crop it square") and a
smaller, faster model does them as well as the largest one. Planning, critique, reference
matching, generative and AI edits, and long multi-part requests go to the main model.
A routine turn that keeps failing its tool calls moves up to the main model mid-turn.
"""

from __future__ import annotations

import re

from photo_agent.advisor import Tier

DEEP_WORDS = re.compile(
    r"\b(remove|erase|get rid|replace|swap|add|put|insert|generate|expand|extend|relight|"
    r"light(ing)? from|restyle|painting|watercolor|anime|cartoon|background|sky|sunset|"
    r"restore|colori[sz]e|old photo|retouch|skin|face|teeth|eyes|cut ?out|subject|person|"
    r"people|object|option|variation|take|plan|why|what do you think|critique|feedback|"
    r"advice|look like|reference|match|style|usual|recipe|straighten|perspective|horizon)\b",
    re.IGNORECASE,
)
"""Words that suggest masks, models, generation, or judgment rather than slider moves."""

MAX_ROUTINE_WORDS = 25
MAX_ROUTINE_PARTS = 3


def choose_tier(
    request: str, *, plan_approved: bool = False, shared_references: bool = False
) -> Tier:
    """The tier for a chat turn: "routine" for simple slider-style requests."""
    if plan_approved or shared_references:
        return "deep"
    words = request.split()
    if len(words) > MAX_ROUTINE_WORDS:
        return "deep"
    parts = re.split(r",|;|\band\b|\bthen\b|\balso\b", request, flags=re.IGNORECASE)
    if len([p for p in parts if p.strip()]) > MAX_ROUTINE_PARTS:
        return "deep"
    if DEEP_WORDS.search(request):
        return "deep"
    return "routine"
