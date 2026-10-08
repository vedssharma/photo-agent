"""Small builders shared by the tests."""

from photo_agent.graph import Step
from photo_agent.layers import EditState
from photo_agent.operations import OpBase


def step(label: str, *ops: OpBase) -> Step:
    """A step whose framing ops go to the framing and the rest into one layer."""
    return Step(label=label, state=EditState.from_operations(list(ops), label))  # type: ignore[arg-type]
