"""Photo critique: what is working in a photo and what is not, in friendly plain words.

Each point covers one aspect (composition, exposure, color, ...) and, when something could
be better, carries a one-click fix: a short request the chat agent can carry out ("Brighten
the shadows a little so the faces are easier to see"). Claude writes the critique when an
API key is set; otherwise a built-in critique covers what measurements can tell
(brightness, clipping, contrast, color cast, saturation), though not composition.
"""

from __future__ import annotations

from typing import Any, Literal

import cv2
import numpy as np
from pydantic import BaseModel, Field

from photo_agent import diagnostics
from photo_agent.advisor import Advisor
from photo_agent.imaging import Array
from photo_agent.render import luma

Aspect = Literal["composition", "exposure", "color", "sharpness", "subject", "mood"]
ASPECTS: tuple[str, ...] = Aspect.__args__  # type: ignore[attr-defined]
MAX_POINTS = 6


class CritiquePoint(BaseModel):
    aspect: Aspect
    verdict: Literal["good", "improve"]
    text: str = Field(description="One or two friendly sentences, no jargon.")
    fix: str | None = Field(
        None, description="For points to improve: a request the agent can carry out."
    )
    fix_label: str | None = Field(None, description="A short button label for the fix.")


class Critique(BaseModel):
    summary: str = Field(description="One sentence on the photo as a whole.")
    points: list[CritiquePoint]
    source: Literal["claude", "built-in"] = "claude"

    def describe(self) -> str:
        lines = [self.summary]
        for p in self.points:
            fix = f" (suggested fix: {p.fix})" if p.fix else ""
            lines.append(f"- {p.aspect}, {p.verdict}: {p.text}{fix}")
        return "\n".join(lines)


SYSTEM = """You are the photo editor inside photo-agent, giving feedback to someone who is \
not an editing expert. Look at the photo as it is now (with any edits applied) and say what \
is working and what is not: composition, exposure, color, sharpness, the subject, and the \
mood. Be warm, specific to this photo, and honest; use everyday words, no jargon \
(say "the bright sky has lost its detail", not "highlights are clipped").

Give two to six points. Lead with what works. For each point that could be better, add a \
fix: a short, concrete request the editing agent can carry out on this photo (it can crop, \
straighten, adjust light and color, select the sky or subject, remove distractions, \
retouch portraits, and more), written as the person would ask it, plus a two to four word \
button label. Points that are fine as they are get no fix. Do not suggest a fix for \
something editing cannot change (like the moment the shutter was pressed)."""

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "points": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "aspect": {"type": "string", "enum": list(ASPECTS)},
                    "verdict": {"type": "string", "enum": ["good", "improve"]},
                    "text": {"type": "string"},
                    "fix": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                    "fix_label": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                },
                "required": ["aspect", "verdict", "text", "fix", "fix_label"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["summary", "points"],
    "additionalProperties": False,
}


def _tidy(critique: Critique) -> Critique:
    points = []
    for p in critique.points[:MAX_POINTS]:
        fix = (p.fix or "").strip() or None
        if p.verdict == "good":
            fix = None
        label = ((p.fix_label or "").strip() or "Fix it") if fix else None
        points.append(p.model_copy(update={"fix": fix, "fix_label": label and label[:40]}))
    return critique.model_copy(update={"points": points})


async def ask_claude(
    advisor: Advisor, image_block: dict[str, Any], measurements: diagnostics.Measurements
) -> Critique:
    answer = await advisor.ask(
        system=SYSTEM,
        content=[
            image_block,  # type: ignore[list-item]
            {
                "type": "text",
                "text": "Measurements of the photo as it is now: "
                f"{measurements.clipped_highlights:.1%} blown-out highlights, "
                f"{measurements.crushed_shadows:.1%} pure black, "
                f"{measurements.oversaturated:.1%} maximally saturated, "
                f"average brightness {measurements.mean_brightness:.2f} (0 to 1).\n\n"
                "What is working and what is not?",
            },
        ],
        schema=SCHEMA,
        tier="deep",
    )
    return _tidy(Critique.model_validate({**answer, "source": "claude"}))


def built_in(pixels: Array) -> Critique:
    """A critique from measurements alone, for when Claude is not available."""
    x = np.clip(pixels, 0.0, 1.0)
    m = diagnostics.measure(x)
    spread = float(np.std(luma(x)))
    lab = cv2.cvtColor(x.astype(np.float32), cv2.COLOR_RGB2LAB)
    cast_a, cast_b = float(lab[..., 1].mean()), float(lab[..., 2].mean())
    points: list[CritiquePoint] = []

    if m.mean_brightness < 0.28:
        points.append(
            CritiquePoint(
                aspect="exposure",
                verdict="improve",
                text="The photo is on the dark side, so details in the shadows are hard to see.",
                fix="Brighten the photo, especially the shadows, without washing it out",
                fix_label="Brighten it",
            )
        )
    elif m.mean_brightness > 0.7:
        points.append(
            CritiquePoint(
                aspect="exposure",
                verdict="improve",
                text="The photo is very bright overall, which makes it look a little washed out.",
                fix="Bring the brightness down a little and add some depth",
                fix_label="Tone it down",
            )
        )
    else:
        points.append(
            CritiquePoint(
                aspect="exposure",
                verdict="good",
                text="The overall brightness is well balanced.",
            )
        )
    if m.clipped_highlights > 0.03:
        points.append(
            CritiquePoint(
                aspect="exposure",
                verdict="improve",
                text="Some of the brightest areas have turned pure white and lost their detail.",
                fix="Recover detail in the brightest areas",
                fix_label="Recover highlights",
            )
        )
    if m.crushed_shadows > 0.03:
        points.append(
            CritiquePoint(
                aspect="exposure",
                verdict="improve",
                text="Parts of the darkest areas have gone completely black.",
                fix="Lift the darkest shadows a little so their detail shows",
                fix_label="Open the shadows",
            )
        )
    if spread < 0.15:
        points.append(
            CritiquePoint(
                aspect="mood",
                verdict="improve",
                text="It looks a little flat; a touch more contrast would give it some punch.",
                fix="Add a little contrast and clarity so it pops",
                fix_label="Add punch",
            )
        )
    elif spread > 0.3:
        points.append(
            CritiquePoint(
                aspect="mood", verdict="good", text="Strong contrast gives it a bold, crisp feel."
            )
        )
    if cast_b > 12 or cast_b < -10 or abs(cast_a) > 10:
        warm = cast_b > 12
        points.append(
            CritiquePoint(
                aspect="color",
                verdict="improve",
                text="There is a "
                + ("strong warm, yellow" if warm else "noticeable color")
                + " tint over everything, so whites and skin may not look natural.",
                fix="Neutralize the color cast so whites look white",
                fix_label="Fix the colors",
            )
        )
    elif m.oversaturated > 0.05:
        points.append(
            CritiquePoint(
                aspect="color",
                verdict="improve",
                text="Some colors are so intense they look a little artificial.",
                fix="Tone down the strongest colors so they look natural",
                fix_label="Calm the colors",
            )
        )
    else:
        points.append(
            CritiquePoint(aspect="color", verdict="good", text="The colors look natural.")
        )
    improve = sum(p.verdict == "improve" for p in points)
    summary = (
        "A solid photo; the basics are in good shape."
        if improve == 0
        else f"A good start, with {improve} thing{'s' if improve > 1 else ''} worth fixing."
    )
    return Critique(summary=summary, points=points[:MAX_POINTS], source="built-in")
