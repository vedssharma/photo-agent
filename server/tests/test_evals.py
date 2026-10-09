import asyncio
from typing import Any

import numpy as np
import pytest

from photo_agent import evals
from photo_agent.evals import CASES, CHECKS, Harness, Outcome, Report, ScriptedModel
from photo_agent.graph import Document


def test_every_case_uses_known_checks_and_fixtures() -> None:
    assert len({c.id for c in CASES}) == len(CASES)
    for case in CASES:
        assert set(case.checks) <= set(CHECKS)
        assert (evals.PHOTOS / case.photo).is_file()


def test_reference_edits_pass_their_own_checks() -> None:
    report = asyncio.run(
        Harness(lambda case: ScriptedModel(case.reference)).run(CASES, "replay", "reference")
    )
    assert report.score == 1.0, evals.markdown(report)


def test_a_bad_edit_scores_low() -> None:
    warm = next(c for c in CASES if c.id == "warmer")
    bad: list[list[dict[str, Any]]] = [
        [{"name": "white_balance", "input": {"temperature": -40}}],
        [{"text": "Done."}],
    ]
    result = asyncio.run(Harness(lambda case: ScriptedModel(bad)).run_case(warm))
    assert result.score < 0.7
    assert result.checks[0].name == "warmer" and result.checks[0].score == 0


def test_a_crashing_case_scores_zero() -> None:
    class Broken:
        async def create(self, **kwargs: Any) -> Any:
            raise RuntimeError("the API is down")

    result = asyncio.run(Harness(lambda case: Broken()).run_case(CASES[0]))
    assert result.score == 0 and result.error == "the API is down"


def outcome(before: Any, after: Any) -> Outcome:
    doc = Document(filename="x", format="PNG", width=1, height=1)
    return Outcome(before=before, after=after, doc=doc, reply="")


@pytest.mark.parametrize(
    ("check", "change", "passes"),
    [
        ("brighter", lambda x: x + 0.1, True),
        ("brighter", lambda x: x - 0.1, False),
        ("monochrome", lambda x: x.mean(axis=2, keepdims=True).repeat(3, axis=2), True),
        ("subtle", lambda x: x * 0.5, False),
        ("aspect_4_5", lambda x: x[:, :20], False),
    ],
)
def test_checks(check: str, change: Any, passes: bool) -> None:
    rng = np.random.default_rng(0)
    before = rng.uniform(0.2, 0.7, (40, 40, 3)).astype(np.float32)
    after = np.clip(change(before), 0, 1).astype(np.float32)
    score, _ = CHECKS[check](outcome(before, after))
    assert (score >= 0.99) is passes


def test_regressions_against_the_baseline() -> None:
    report = Report.model_validate(
        {
            "mode": "live",
            "model": "m",
            "score": 0.7,
            "cases": [
                {"id": "a", "request": "", "score": 0.3, "checks": [], "reply": "", "seconds": 1}
            ],
        }
    )
    assert evals.compare(report, {"score": 0.72, "cases": {"a": 0.6}}) == []
    found = evals.compare(report, {"score": 0.8, "cases": {"a": 1.0}})
    assert found == ["suite 70% < baseline 80%", "a 30% < baseline 100%"]


def test_main_replay_exits_cleanly(capsys: pytest.CaptureFixture[str]) -> None:
    assert evals.main(["--replay", "--case", "warmer"]) == 0
    assert "Agent evals (replay" in capsys.readouterr().out
