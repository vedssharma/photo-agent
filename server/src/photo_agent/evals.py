"""Agent evals: a fixed set of photos and requests with scored results, so changes to the
agent's prompt, tools, or model do not silently make its edits worse.

Each case opens a fixture photo, sends one request through the real agent loop (the same
`AgentService` the chat uses), renders the result, and scores it with checks that
measure what the request promised: "warmer" must move the color toward yellow, "crop it
for Instagram" must come out 4:5, nothing may blow out the highlights, a multi-step
generative request must show a plan first. Checks score 0 to 1; a case scores their mean
and the suite the mean of its cases.

Two ways to run it:

    uv run python -m photo_agent.evals             # live: Claude edits (needs a key)
    uv run python -m photo_agent.evals --replay    # each case's scripted reference edit

Replay runs offline and checks that the harness and the checks themselves work (the
reference edits must pass); CI runs it on every change to the agent. With an API key, CI
also runs the live suite and fails when it scores below `evals/baseline.json`. Add
`--judge` to a live run to have Claude also rate each before/after pair from 1 to 10.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import tempfile
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
from anthropic.types.beta import BetaMessage, BetaMessageParam, BetaToolParam
from pydantic import BaseModel

from photo_agent import diagnostics, imaging
from photo_agent.advisor import Advisor, ClaudeAdvisor
from photo_agent.agent import AgentEvent, AgentService, ClaudeModel, ModelClient, image_block
from photo_agent.graph import Document
from photo_agent.render import RenderCache, luma
from photo_agent.settings import REPO_ROOT, Settings
from photo_agent.store import DocumentStore
from photo_agent.vision.backends import WorkerConfig
from photo_agent.vision.worker import ModelWorker

PHOTOS = REPO_ROOT / "fixtures" / "photos"
BASELINE = REPO_ROOT / "server" / "evals" / "baseline.json"
TOLERANCE = 0.05
"""How far the live suite may score below the baseline before CI fails."""


@dataclass
class Outcome:
    """What a case produced, for the checks to measure."""

    before: imaging.Array
    after: imaging.Array
    doc: Document
    reply: str

    @property
    def changed(self) -> bool:
        return self.doc.head is not None


Check = Callable[[Outcome], tuple[float, str]]


def _lab(x: imaging.Array) -> imaging.Array:
    small = imaging.resize_long_edge(np.clip(x, 0, 1), 256).astype(np.float32)
    return cast(imaging.Array, cv2.cvtColor(small, cv2.COLOR_RGB2LAB))


def _ramp(value: float, full: float) -> float:
    """0 at or below 0, 1 at or above `full`, linear between."""
    return float(min(1.0, max(0.0, value / full)))


def brighter(by: float = 0.03) -> Check:
    def check(o: Outcome) -> tuple[float, str]:
        delta = float(luma(o.after).mean() - luma(o.before).mean())
        return _ramp(delta, by), f"brightness {delta:+.3f}"

    return check


def darker(by: float = 0.02) -> Check:
    def check(o: Outcome) -> tuple[float, str]:
        delta = float(luma(o.before).mean() - luma(o.after).mean())
        return _ramp(delta, by), f"darkened by {delta:+.3f}"

    return check


def warmer(by: float = 2.0) -> Check:
    def check(o: Outcome) -> tuple[float, str]:
        delta = float(_lab(o.after)[..., 2].mean() - _lab(o.before)[..., 2].mean())
        return _ramp(delta, by), f"yellow-blue {delta:+.2f}"

    return check


def cooler(by: float = 2.0) -> Check:
    def check(o: Outcome) -> tuple[float, str]:
        delta = float(_lab(o.before)[..., 2].mean() - _lab(o.after)[..., 2].mean())
        return _ramp(delta, by), f"cooled by {delta:+.2f}"

    return check


def _chroma(x: imaging.Array) -> float:
    lab = _lab(x)
    return float(np.hypot(lab[..., 1], lab[..., 2]).mean())


def more_color(by: float = 1.5) -> Check:
    def check(o: Outcome) -> tuple[float, str]:
        delta = _chroma(o.after) - _chroma(o.before)
        return _ramp(delta, by), f"colorfulness {delta:+.2f}"

    return check


def monochrome(at_most: float = 2.0) -> Check:
    def check(o: Outcome) -> tuple[float, str]:
        chroma = _chroma(o.after)
        return (1.0 if chroma <= at_most else 0.0), f"colorfulness {chroma:.2f}"

    return check


def more_contrast(by: float = 0.006) -> Check:
    def check(o: Outcome) -> tuple[float, str]:
        delta = float(luma(o.after).std() - luma(o.before).std())
        return _ramp(delta, by), f"contrast {delta:+.3f}"

    return check


def aspect(ratio: str) -> Check:
    w, h = (float(v) for v in ratio.split(":"))

    def check(o: Outcome) -> tuple[float, str]:
        got = o.after.shape[1] / o.after.shape[0]
        off = abs(math.log(got / (w / h)))
        return (1.0 if off < 0.02 else 0.0), f"aspect {got:.3f} (want {w / h:.3f})"

    return check


def subtle(at_most: float = 0.06) -> Check:
    def check(o: Outcome) -> tuple[float, str]:
        if o.after.shape != o.before.shape:
            return 0.0, "framing changed"
        diff = float(np.abs(o.after - o.before).mean())
        return (1.0 if diff <= at_most else max(0.0, 1 - (diff - at_most) / at_most)), (
            f"mean change {diff:.3f}"
        )

    return check


def no_overshoot(o: Outcome) -> tuple[float, str]:
    problems = diagnostics.warnings(diagnostics.measure(o.before), diagnostics.measure(o.after))
    return (0.0 if problems else 1.0), "; ".join(problems) or "no overshoots"


def changed(o: Outcome) -> tuple[float, str]:
    return (1.0 if o.changed else 0.0), "edited" if o.changed else "nothing changed"


def unchanged(o: Outcome) -> tuple[float, str]:
    return (0.0 if o.changed else 1.0), "edited" if o.changed else "left as is"


def named_layers(o: Outcome) -> tuple[float, str]:
    layers = o.doc.state.layers
    if not layers:
        return 0.0, "no layers"
    request = o.doc.steps[-1].request if o.doc.steps else ""
    named = [lay for lay in layers if lay.name.rstrip("…").lower() not in request.lower()]
    return len(named) / len(layers), f"{len(named)} of {len(layers)} layers named for the change"


def plan_first(o: Outcome) -> tuple[float, str]:
    plan = o.doc.pending_plan
    if plan is None:
        return 0.0, "no plan shown"
    return (1.0 if not o.changed else 0.5), f"plan with {len(plan.steps)} steps"


class Case(BaseModel):
    id: str
    photo: str
    request: str
    checks: list[str]
    """Names of checks, from CHECKS."""
    reference: list[list[dict[str, Any]]]
    """A known-good edit for replay mode: model rounds, each a list of tool calls
    ({"name", "input"}) or a {"text"} reply."""


CHECKS: dict[str, Check] = {
    "brighter": brighter(),
    "darker": darker(),
    "warmer": warmer(),
    "cooler": cooler(),
    "more_color": more_color(),
    "monochrome": monochrome(),
    "more_contrast": more_contrast(),
    "aspect_4_5": aspect("4:5"),
    "aspect_16_9": aspect("16:9"),
    "subtle": subtle(),
    "no_overshoot": no_overshoot,
    "changed": changed,
    "unchanged": unchanged,
    "named_layers": named_layers,
    "plan_first": plan_first,
}


def _tool(tool_name: str, /, **args: Any) -> dict[str, Any]:
    return {"name": tool_name, "input": args}


def _say(text: str) -> list[dict[str, Any]]:
    return [{"text": text}]


CASES: list[Case] = [
    Case(
        id="dark-brighten",
        photo="low-light.jpg",
        request="This is too dark, brighten it up",
        checks=["brighter", "no_overshoot", "named_layers"],
        reference=[
            [
                _tool("add_layer", name="Brighter"),
                _tool("exposure", stops=0.7),
                _tool("saturation", amount=-15),
            ],
            _say("Brightened it and opened up the shadows."),
        ],
    ),
    Case(
        id="warmer",
        photo="landscape.png",
        request="Make it warmer",
        checks=["warmer", "no_overshoot", "subtle"],
        reference=[
            [_tool("add_layer", name="Warmer tones"), _tool("white_balance", temperature=30)],
            _say("Warmed it up."),
        ],
    ),
    Case(
        id="cool-moody",
        photo="portrait.jpg",
        request="Cooler and moodier please",
        checks=["cooler", "darker", "no_overshoot"],
        reference=[
            [
                _tool("add_layer", name="Cool and moody"),
                _tool("white_balance", temperature=-30),
                _tool("exposure", stops=-0.3),
            ],
            _say("Cooled the colors and darkened it a touch."),
        ],
    ),
    Case(
        id="black-and-white",
        photo="landscape.png",
        request="Black and white, with some punch",
        checks=["monochrome", "more_contrast", "named_layers"],
        reference=[
            [
                _tool("add_layer", name="Punchy black and white"),
                _tool("saturation", amount=-100),
                _tool("contrast", amount=40),
            ],
            _say("Made it black and white with more contrast."),
        ],
    ),
    Case(
        id="instagram-crop",
        photo="portrait.jpg",
        request="Crop it for an Instagram post",
        checks=["aspect_4_5", "changed"],
        reference=[[_tool("crop", aspect="4:5")], _say("Cropped to 4:5 for the feed.")],
    ),
    Case(
        id="widescreen-crop",
        photo="landscape.png",
        request="Crop it to 16:9 for my desktop",
        checks=["aspect_16_9", "changed"],
        reference=[[_tool("crop", aspect="16:9")], _say("Cropped to 16:9.")],
    ),
    Case(
        id="touch-more-color",
        photo="landscape.png",
        request="Just a touch more color, keep it natural",
        checks=["more_color", "subtle", "no_overshoot"],
        reference=[
            [_tool("add_layer", name="A touch more color"), _tool("vibrance", amount=20)],
            _say("Added a little color."),
        ],
    ),
    Case(
        id="make-it-pop",
        photo="portrait.jpg",
        request="Make it pop a little more",
        checks=["more_contrast", "no_overshoot", "named_layers"],
        reference=[
            [
                _tool("add_layer", name="More pop"),
                _tool("contrast", amount=20),
                _tool("saturation", amount=-10),
            ],
            _say("Added contrast, keeping the colors in check."),
        ],
    ),
    Case(
        id="just-a-question",
        photo="low-light.jpg",
        request="What do you think of this photo? Don't change anything yet.",
        checks=["unchanged"],
        reference=[_say("It's moody but quite dark; brightening the shadows would help.")],
    ),
    Case(
        id="plan-before-generating",
        photo="landscape.png",
        request="Replace the sky with a sunset, expand it to 16:9, and relight it to match",
        checks=["plan_first"],
        reference=[
            [
                _tool(
                    "propose_plan",
                    steps=[
                        {"text": "Replace the sky with a sunset", "kind": "generative"},
                        {"text": "Expand the canvas to 16:9", "kind": "generative"},
                        {"text": "Relight it with warm sunset light", "kind": "generative"},
                    ],
                )
            ]
        ],
    ),
]


class ScriptedModel:
    """Plays back a case's reference edit as if Claude made it."""

    def __init__(self, rounds: Sequence[Sequence[dict[str, Any]]]) -> None:
        self.rounds = list(rounds)

    async def create(
        self,
        *,
        system: str,
        tools: Sequence[BetaToolParam],
        messages: Sequence[BetaMessageParam],
        on_text: Callable[[str], Awaitable[None]],
    ) -> BetaMessage:
        blocks: list[dict[str, Any]] = []
        for i, item in enumerate(self.rounds.pop(0) if self.rounds else []):
            if "text" in item:
                await on_text(item["text"])
                blocks.append({"type": "text", "text": item["text"], "citations": None})
            else:
                blocks.append({"type": "tool_use", "id": f"toolu_{i}", **item})
        return BetaMessage.model_validate(
            {
                "id": "msg_replay",
                "type": "message",
                "role": "assistant",
                "model": "replay",
                "content": blocks,
                "stop_reason": "tool_use" if any("name" in b for b in blocks) else "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            }
        )


class CheckResult(BaseModel):
    name: str
    score: float
    note: str


class CaseResult(BaseModel):
    id: str
    request: str
    score: float
    checks: list[CheckResult]
    reply: str
    seconds: float
    judge: float | None = None
    error: str | None = None


class Report(BaseModel):
    mode: str
    model: str
    score: float
    cases: list[CaseResult]


JUDGE_SYSTEM = """You judge a photo editor's work. You see the photo before and after, and \
what the person asked for. Rate the edit from 1 to 10: does it do what was asked, does it \
look professional and natural, and is it free of artifacts like blown highlights, garish \
color, or odd casts? 5 is passable, 8 is what a skilled editor would deliver."""

JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"score": {"type": "integer"}, "reason": {"type": "string"}},
    "required": ["score", "reason"],
    "additionalProperties": False,
}


@dataclass
class Harness:
    model_for: Callable[[Case], ModelClient]
    judge: Advisor | None = None
    data_dir: Path = field(default_factory=lambda: Path(tempfile.mkdtemp(prefix="evals-")))

    def __post_init__(self) -> None:
        worker = ModelWorker(WorkerConfig(backends="classical"), "inline")
        self.store = DocumentStore(self.data_dir, worker=worker, backends="classical")
        self.renders = RenderCache()

    async def run_case(self, case: Case) -> CaseResult:
        started = time.monotonic()
        doc = self.store.create(case.photo, (PHOTOS / case.photo).read_bytes())
        loaded = self.store.image(doc.id)
        before = self.renders.get_or_render(doc.id, loaded.proxy, doc.state, loaded.proxy_context)

        async def ignore(event: AgentEvent) -> None:
            pass

        service = AgentService(self.store, self.renders, self.model_for(case))
        try:
            doc = await service.run_turn(doc.id, case.request, ignore)
        except Exception as exc:  # a broken case scores 0 rather than stopping the suite
            return CaseResult(
                id=case.id,
                request=case.request,
                score=0.0,
                checks=[],
                reply="",
                seconds=time.monotonic() - started,
                error=str(exc),
            )
        after = self.renders.get_or_render(doc.id, loaded.proxy, doc.state, loaded.proxy_context)
        reply = doc.chat[-1].text if doc.chat else ""
        outcome = Outcome(before=before, after=after, doc=doc, reply=reply)
        checks = []
        for name in case.checks:
            score, note = CHECKS[name](outcome)
            checks.append(CheckResult(name=name, score=round(score, 3), note=note))
        judge = await self._judge(case, outcome) if self.judge and outcome.changed else None
        return CaseResult(
            id=case.id,
            request=case.request,
            score=round(sum(c.score for c in checks) / len(checks), 3),
            checks=checks,
            reply=reply,
            seconds=round(time.monotonic() - started, 2),
            judge=judge,
        )

    async def _judge(self, case: Case, outcome: Outcome) -> float | None:
        assert self.judge is not None
        try:
            answer = await self.judge.ask(
                system=JUDGE_SYSTEM,
                content=[
                    {"type": "text", "text": "Before:"},
                    image_block(outcome.before),
                    {"type": "text", "text": "After:"},
                    image_block(outcome.after),
                    {"type": "text", "text": f"The person asked: {case.request}"},
                ],
                schema=JUDGE_SCHEMA,
                tier="routine",
            )
        except Exception:
            return None
        return float(min(10, max(1, int(answer["score"]))))

    async def run(self, cases: Sequence[Case], mode: str, model: str) -> Report:
        results = [await self.run_case(case) for case in cases]
        score = round(sum(r.score for r in results) / len(results), 3) if results else 0.0
        return Report(mode=mode, model=model, score=score, cases=results)


def markdown(report: Report, baseline: dict[str, Any] | None = None) -> str:
    lines = [
        f"## Agent evals ({report.mode}, {report.model}): {report.score:.0%}",
        "",
        "| Case | Score | Judge | Checks |",
        "| --- | --- | --- | --- |",
    ]
    for case in report.cases:
        notes = case.error or "; ".join(f"{c.name} {c.score:.0%} ({c.note})" for c in case.checks)
        judge = f"{case.judge:g}/10" if case.judge is not None else ""
        lines.append(f"| {case.id} | {case.score:.0%} | {judge} | {notes} |")
    if baseline:
        lines += ["", f"Baseline: {baseline['score']:.0%} (tolerance {TOLERANCE:.0%})."]
    return "\n".join(lines)


def compare(report: Report, baseline: dict[str, Any]) -> list[str]:
    """Regressions against the baseline: the suite or a case scoring clearly lower."""
    found = []
    if report.score < baseline["score"] - TOLERANCE:
        found.append(f"suite {report.score:.0%} < baseline {baseline['score']:.0%}")
    for case in report.cases:
        floor = baseline.get("cases", {}).get(case.id)
        if floor is not None and case.score < floor - 0.34:
            found.append(f"{case.id} {case.score:.0%} < baseline {floor:.0%}")
    return found


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m photo_agent.evals", description=__doc__)
    parser.add_argument("--replay", action="store_true", help="play the reference edits")
    parser.add_argument("--judge", action="store_true", help="have Claude rate each result")
    parser.add_argument("--case", action="append", help="run only these case ids")
    parser.add_argument("--report", type=Path, help="write the JSON report here")
    parser.add_argument("--update-baseline", action="store_true", help="save the scores")
    args = parser.parse_args(argv)

    cases = [c for c in CASES if not args.case or c.id in args.case]
    settings = Settings()
    key = settings.anthropic_api_key
    if args.replay:
        mode, model_name = "replay", "reference edits"

        def model_for(case: Case) -> ModelClient:
            return ScriptedModel(case.reference)

        judge = None
    else:
        if key is None:
            print("Live evals need ANTHROPIC_API_KEY; use --replay to run offline.")
            return 2
        mode, model_name = "live", settings.anthropic_model
        live = ClaudeModel(key.get_secret_value(), settings.anthropic_model)

        def model_for(case: Case) -> ModelClient:
            return live

        judge = ClaudeAdvisor(key.get_secret_value(), settings.anthropic_model)

    harness = Harness(model_for, judge if args.judge else None)
    report = asyncio.run(harness.run(cases, mode, model_name))
    baseline = json.loads(BASELINE.read_text()) if BASELINE.is_file() else None
    summary = markdown(report, baseline if mode == "live" else None)
    print(summary)
    if path := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(path, "a") as out:
            out.write(summary + "\n")
    if args.report:
        args.report.write_text(report.model_dump_json(indent=2))

    if args.replay:
        # The reference edits are known good: anything short of full marks is a broken check.
        broken = [c.id for c in report.cases if c.score < 1.0]
        if broken:
            print(f"Reference edits failed their own checks: {', '.join(broken)}")
            return 1
        return 0
    if args.update_baseline:
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        BASELINE.write_text(
            json.dumps(
                {"score": report.score, "cases": {c.id: c.score for c in report.cases}},
                indent=2,
            )
            + "\n"
        )
        return 0
    if baseline is None:
        return 0
    regressions = compare(report, baseline)
    for line in regressions:
        print(f"Regression: {line}")
    return 1 if regressions else 0


if __name__ == "__main__":
    sys.exit(main())
