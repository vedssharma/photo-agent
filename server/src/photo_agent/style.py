"""Personal style memory: the person's taste, learned from the edits they keep and undo.

Every edit leaves a trace in a few everyday sliders (warmth, contrast, saturation, grain,
...). The app notes what the person did with them:

- kept: a photo was downloaded with these values, or a suggestion was picked;
- adjusted: they set a slider by hand, which is the clearest statement of taste;
- rejected: they undid an agent change, so it went too far in that direction.

From those notes come tendencies ("usually warmer, around +12") that the agent is told
with every request, and "my usual look": one layer with the values they keep coming back
to. Only whole-photo layers count; a masked layer says more about that photo than about
taste. Everything lives in `<data_dir>/style.json`, on this computer, and can be forgotten.
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from photo_agent.layers import EditState, Layer, new_layer_id
from photo_agent.suggestions import slider_operations

MAX_SIGNALS = 500
MIN_SAMPLES = 2
"""How many notes a slider needs before it counts as a tendency."""
FOLD_WINDOW = timedelta(minutes=2)
"""Hand adjustments to the same slider this close together are one decision."""

TRACKED: dict[str, tuple[str, str]] = {
    "exposure": ("exposure", "stops"),
    "contrast": ("contrast", "amount"),
    "highlights": ("highlights", "amount"),
    "shadows": ("shadows", "amount"),
    "whites": ("whites", "amount"),
    "blacks": ("blacks", "amount"),
    "temperature": ("white_balance", "temperature"),
    "tint": ("white_balance", "tint"),
    "vibrance": ("vibrance", "amount"),
    "saturation": ("saturation", "amount"),
    "clarity": ("clarity", "amount"),
    "dehaze": ("dehaze", "amount"),
    "vignette": ("vignette", "amount"),
    "grain": ("grain", "amount"),
}
"""Slider name to (operation, parameter). Names match `suggestions.SLIDERS`."""

PLAIN: dict[str, tuple[str, str]] = {
    "exposure": ("brighter", "darker"),
    "contrast": ("more contrast", "softer contrast"),
    "highlights": ("brighter highlights", "recovered highlights"),
    "shadows": ("lifted shadows", "deeper shadows"),
    "whites": ("brighter whites", "dimmer whites"),
    "blacks": ("faded, matte blacks", "deeper blacks"),
    "temperature": ("warmer", "cooler"),
    "tint": ("more magenta", "more green"),
    "vibrance": ("richer color", "more muted color"),
    "saturation": ("more saturated", "less saturated"),
    "clarity": ("crisper texture", "softer texture"),
    "dehaze": ("clearer, less haze", "hazier, more atmosphere"),
    "vignette": ("lighter edges", "darker edges (vignette)"),
    "grain": ("film grain", "no grain"),
}

USUAL_LOOK_SKIPS = {"exposure"}
"""Sliders left out of "my usual look": how bright to go depends on the photo."""

Verdict = Literal["kept", "adjusted", "rejected"]
WEIGHT: dict[Verdict, float] = {"kept": 1.0, "adjusted": 2.0, "rejected": 0.0}


def now() -> datetime:
    return datetime.now(UTC)


class Signal(BaseModel):
    slider: str
    value: float
    verdict: Verdict
    at: datetime = Field(default_factory=now)


class Tendency(BaseModel):
    slider: str
    preferred: float = Field(description="The value they tend to keep (weighted mean).")
    samples: int
    rejected: list[float] = Field(description="Values they undid, most recent last.")

    def describe(self) -> str:
        step = 0.1 if self.slider == "exposure" else 3
        if abs(self.preferred) < step:
            lean = f"little or no {self.slider}"
        else:
            lean = PLAIN[self.slider][0 if self.preferred > 0 else 1]
        unit = " stops" if self.slider == "exposure" else ""
        return f"{lean} ({self.slider} around {self.preferred:+.2g}{unit}, {self.samples} edits)"


class StyleSummary(BaseModel):
    """What the app has learned about the person's taste."""

    tendencies: list[Tendency]
    lines: list[str] = Field(description="The tendencies in plain words.")
    has_usual_look: bool
    signals: int = Field(description="How many notes it has taken.")


def slider_values(state: EditState) -> dict[str, float]:
    """The tracked sliders' effective values in a state's visible whole-photo layers."""
    totals: dict[str, float] = {}
    for layer in state.layers:
        if not layer.visible or layer.mask is not None or layer.is_content:
            continue
        weight = layer.opacity / 100
        for op in layer.operations:
            for slider, (name, field) in TRACKED.items():
                if op.op == name:
                    value = float(getattr(op, field)) * weight
                    totals[slider] = totals.get(slider, 0.0) + value
    return {k: round(v, 3) for k, v in totals.items() if v != 0}


class Profile(BaseModel):
    signals: list[Signal] = Field(default_factory=list)

    def tendencies(self) -> list[Tendency]:
        found = []
        for slider in TRACKED:
            mine = [s for s in self.signals if s.slider == slider]
            counted = [s for s in mine if WEIGHT[s.verdict] > 0]
            if len(counted) < MIN_SAMPLES:
                continue
            total = sum(WEIGHT[s.verdict] for s in counted)
            preferred = sum(s.value * WEIGHT[s.verdict] for s in counted) / total
            found.append(
                Tendency(
                    slider=slider,
                    preferred=round(preferred, 2),
                    samples=len(counted),
                    rejected=[s.value for s in mine if s.verdict == "rejected"][-3:],
                )
            )
        return found

    def describe(self) -> str | None:
        """The person's taste in plain words for the agent, or None if nothing is known."""
        tendencies = self.tendencies()
        rejected = [
            f"{s.slider} {s.value:+g}"
            for s in self.signals[-40:]
            if s.verdict == "rejected" and s.slider in TRACKED
        ]
        if not tendencies and not rejected:
            return None
        lines = ["The person's taste, learned from edits they kept, set, or undid:"]
        lines += [f"- tends toward {t.describe()}" for t in tendencies]
        if rejected:
            lines.append("- recently undid: " + ", ".join(dict.fromkeys(rejected[-6:])))
        lines.append(
            "Lean toward this taste unless the request says otherwise; it is a starting "
            "point, not a rule."
        )
        return "\n".join(lines)

    def usual_look(self) -> Layer | None:
        """One layer with the values they keep coming back to, or None if too little is
        known yet."""
        values = {
            t.slider: t.preferred
            for t in self.tendencies()
            if t.slider not in USUAL_LOOK_SKIPS and abs(t.preferred) >= 3
        }
        operations = slider_operations(values)
        if not operations:
            return None
        return Layer.model_validate(
            {
                "id": new_layer_id(),
                "name": "My usual look",
                "operations": [op.model_dump() for op in operations],
            }
        )

    def summary(self) -> StyleSummary:
        tendencies = self.tendencies()
        return StyleSummary(
            tendencies=tendencies,
            lines=[t.describe() for t in tendencies],
            has_usual_look=self.usual_look() is not None,
            signals=len(self.signals),
        )


class StyleStore:
    """The style profile on disk. Safe to share between requests."""

    def __init__(self, data_dir: Path) -> None:
        self.path = data_dir / "style.json"
        self._lock = threading.Lock()

    def load(self) -> Profile:
        with self._lock:
            return self._read()

    def _read(self) -> Profile:
        if not self.path.is_file():
            return Profile()
        try:
            return Profile.model_validate_json(self.path.read_text())
        except ValueError:
            return Profile()

    def record(self, values: dict[str, float], verdict: Verdict) -> None:
        if not values:
            return
        with self._lock:
            profile = self._read()
            at = now()
            for slider, value in values.items():
                if slider not in TRACKED:
                    continue
                last = next((s for s in reversed(profile.signals) if s.slider == slider), None)
                if (
                    verdict == "adjusted"
                    and last is not None
                    and last.verdict == "adjusted"
                    and at - last.at < FOLD_WINDOW
                ):
                    profile.signals.remove(last)
                profile.signals.append(Signal(slider=slider, value=value, verdict=verdict, at=at))
            profile.signals = profile.signals[-MAX_SIGNALS:]
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(profile.model_dump_json())
            tmp.replace(self.path)

    def kept(self, state: EditState) -> None:
        """The person kept these edits (downloaded the photo, picked a suggestion)."""
        self.record(slider_values(state), "kept")

    def adjusted(self, before: EditState, after: EditState) -> None:
        """The person changed sliders by hand: note the values they set."""
        old, new = slider_values(before), slider_values(after)
        changed = {k: new.get(k, 0.0) for k in set(old) | set(new) if old.get(k) != new.get(k)}
        self.record(changed, "adjusted")

    def rejected(self, before: EditState, after: EditState) -> None:
        """The person undid an agent change from `before` to `after`: note what it added."""
        old, new = slider_values(before), slider_values(after)
        delta = {k: round(new.get(k, 0.0) - old.get(k, 0.0), 3) for k in set(old) | set(new)}
        self.record({k: v for k, v in delta.items() if v != 0}, "rejected")

    def forget(self) -> None:
        with self._lock:
            self.path.unlink(missing_ok=True)


def apply_usual_look(profile: Profile, state: EditState) -> EditState | None:
    layer = profile.usual_look()
    if layer is None:
        return None
    return state.model_copy(update={"layers": [*state.layers, layer]}, deep=True)
