"""Edit graph: a document is the original image plus a tree of edit steps.

The graph never stores pixels. Each step (an agent turn or a manual tweak) records the full
edit state (framing plus layers, see `photo_agent.layers`) as it stood after that step, and
points at the step it was made from. Undo and redo move `head` along a branch; jumping to an
older step and editing from there starts a new branch without losing the old one. Any step
can be re-rendered at preview or full resolution from the original.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from photo_agent.layers import EditState
from photo_agent.operations import Operation, OperationAdapter


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def now() -> datetime:
    return datetime.now(UTC)


StepKind = Literal["agent", "manual"]


class Step(BaseModel):
    """One named change to the photo and the edit state it left behind."""

    id: str = Field(default_factory=new_id)
    parent: str | None = None
    """The step this one was made from; None for steps made from the original."""
    kind: StepKind = "agent"
    label: str
    """Short name shown in the history panel."""
    request: str = ""
    """For agent steps, what the user asked for."""
    reply: str = ""
    """For agent steps, the agent's plain-language explanation of what it changed."""
    state: EditState = Field(default_factory=EditState)
    """The complete edit state after this step."""
    coalesce: str | None = None
    """For manual steps, what was tweaked (e.g. one slider). Repeated tweaks of the same
    thing fold into one step instead of flooding the history."""
    created_at: datetime = Field(default_factory=now)

    @model_validator(mode="before")
    @classmethod
    def _from_v1(cls, data: Any) -> Any:
        """Phase 1 steps held a flat operation list; it becomes framing plus one layer."""
        if isinstance(data, dict) and "operations" in data:
            data = {**data}
            ops = data.pop("operations")
            name = str(data.get("label") or data.get("request") or "Edits")
            data.setdefault("state", EditState.from_operations(_validate_ops(ops), name))
        return data


def _validate_ops(raw: Any) -> list[Operation]:
    return [OperationAdapter.validate_python(op) for op in raw]


class PlanStep(BaseModel):
    text: str = Field(min_length=1, max_length=300, description="What this step does.")
    kind: Literal["adjust", "ai", "generative"] = Field(
        "adjust",
        description="adjust: quick slider-style edits; ai: runs an AI model to select, "
        "remove, or retouch; generative: paints new pixels with an image generation model.",
    )


class Plan(BaseModel):
    """What the agent means to do for a multi-step request, shown before the slow or
    generative steps run, so the person can approve or change it."""

    id: str = Field(default_factory=new_id)
    steps: list[PlanStep] = Field(min_length=1, max_length=12)

    def describe(self) -> str:
        return "\n".join(f"{i}. {step.text}" for i, step in enumerate(self.steps, start=1))


class ChatEntry(BaseModel):
    """One line of the conversation shown in the chat panel and replayed to the agent.

    The conversation is linear even when the history branches: undo, redo, and jumps are
    recorded as `event` entries, so the agent knows its earlier change is no longer in effect.
    """

    role: Literal["user", "assistant", "event"]
    text: str
    step_id: str | None = None
    """The step this entry produced, for assistant replies that changed the photo."""
    plan: Plan | None = None
    """For assistant replies that propose a plan instead of carrying it out yet."""
    created_at: datetime = Field(default_factory=now)

    @model_validator(mode="before")
    @classmethod
    def _from_v1(cls, data: Any) -> Any:
        if isinstance(data, dict) and "turn_id" in data:
            data = {**data}
            data.setdefault("step_id", data.pop("turn_id"))
        return data


class Document(BaseModel):
    """A photo being edited. Stored as JSON next to the original file."""

    id: str = Field(default_factory=new_id)
    filename: str
    format: str
    """Source format: JPEG, PNG, or HEIF."""
    width: int
    height: int
    """Upright size of the original, in pixels."""
    created_at: datetime = Field(default_factory=now)
    steps: list[Step] = Field(default_factory=list)
    """Every step ever made, in the order they were made. Parents come before children."""
    head: str | None = None
    """The step currently shown, or None for the original."""
    tip: str | None = None
    """The newest step on the current branch; redo walks from head toward it."""
    chat: list[ChatEntry] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _from_v1(cls, data: Any) -> Any:
        """Read Phase 1 documents, whose history was a flat list of turns and a cursor."""
        if not isinstance(data, dict) or "turns" not in data:
            return data
        data = {**data}
        turns = data.pop("turns")
        cursor = data.pop("cursor", len(turns))
        steps, parent = [], None
        for turn in turns:
            steps.append({**turn, "label": turn["request"], "parent": parent, "kind": "agent"})
            parent = turn["id"]
        data["steps"] = steps
        data["head"] = turns[cursor - 1]["id"] if cursor else None
        data["tip"] = turns[-1]["id"] if turns else None
        return data

    @property
    def pending_plan(self) -> Plan | None:
        """A plan the agent proposed in its last reply, waiting for the person's go-ahead."""
        for entry in reversed(self.chat):
            if entry.role == "user":
                return None
            if entry.role == "assistant":
                return entry.plan
        return None

    def step(self, step_id: str) -> Step:
        for s in self.steps:
            if s.id == step_id:
                return s
        raise KeyError(step_id)

    def lineage(self, step_id: str | None) -> list[Step]:
        """The steps from the original down to `step_id`, inclusive."""
        chain: list[Step] = []
        while step_id is not None:
            s = self.step(step_id)
            chain.append(s)
            step_id = s.parent
        return chain[::-1]

    @property
    def current(self) -> Step | None:
        return self.step(self.head) if self.head else None

    @property
    def state(self) -> EditState:
        """The edits currently in effect (a copy; commit a step to change them)."""
        current = self.current
        return current.state.model_copy(deep=True) if current else EditState()

    @property
    def revision(self) -> str:
        """Identifies what the current edits render to; changes whenever the result would."""
        return "original" if self.head is None else self.state.fingerprint

    @property
    def can_undo(self) -> bool:
        return self.head is not None

    @property
    def can_redo(self) -> bool:
        return self.tip != self.head

    def _next_toward_tip(self) -> Step | None:
        if self.tip == self.head:
            return None
        path = self.lineage(self.tip)
        index = next(i for i, s in enumerate(path) if s.id == self.head) + 1 if self.head else 0
        return path[index]

    def commit(self, step: Step) -> None:
        """Add a step on top of the current one. Anything after the current step stays in the
        history as another branch."""
        step.parent = self.head
        self.steps.append(step)
        self.head = self.tip = step.id

    def replace_head(self, step: Step) -> None:
        """Swap the current step for an updated version of itself (same id and parent)."""
        index = next(i for i, s in enumerate(self.steps) if s.id == self.head)
        step.id, step.parent = self.steps[index].id, self.steps[index].parent
        self.steps[index] = step

    def edit_by_hand(self, label: str, state: EditState, coalesce: str | None = None) -> bool:
        """Record a manual change as a named step. Returns False if nothing changed.

        Successive tweaks with the same `coalesce` key (dragging one slider several times)
        update the step they started instead of adding one step per tweak.
        """
        if state == self.state:
            return False
        step = Step(kind="manual", label=label, state=state, coalesce=coalesce)
        current = self.current
        text = f"Edited by hand: {label}"
        if (
            coalesce
            and current is not None
            and current.kind == "manual"
            and current.coalesce == coalesce
            and not self.has_children(current.id)
        ):
            self.replace_head(step)
            if self.chat and self.chat[-1].step_id == step.id:
                self.chat[-1].text = text
            return True
        self.commit(step)
        self.chat.append(ChatEntry(role="event", text=text, step_id=step.id))
        return True

    def has_children(self, step_id: str) -> bool:
        return any(s.parent == step_id for s in self.steps)

    def undo(self) -> bool:
        current = self.current
        if current is None:
            return False
        self.head = current.parent
        self.chat.append(ChatEntry(role="event", text=f"Undid: {current.label}"))
        return True

    def redo(self) -> bool:
        following = self._next_toward_tip()
        if following is None:
            return False
        self.head = following.id
        self.chat.append(ChatEntry(role="event", text=f"Redid: {following.label}"))
        return True

    def checkout(self, step_id: str | None) -> bool:
        """Show any step in the history (None for the original). Raises KeyError."""
        if step_id == self.head:
            return False
        if step_id is not None and step_id not in {s.id for s in self.lineage(self.tip)}:
            # Another branch: redo should follow it to its newest step.
            self.tip = self._newest_leaf(step_id)
        self.head = step_id
        label = self.step(step_id).label if step_id else "the original photo"
        self.chat.append(ChatEntry(role="event", text=f"Jumped in history to: {label}"))
        return True

    def _newest_leaf(self, step_id: str) -> str:
        while True:
            children = [s for s in self.steps if s.parent == step_id]
            if not children:
                return step_id
            step_id = max(children, key=lambda s: s.created_at).id


class StepView(BaseModel):
    id: str
    parent: str | None
    kind: StepKind
    label: str
    created_at: datetime
    active: bool
    """Whether this step's edits are part of what is shown (it is the head or before it)."""


class DocumentView(BaseModel):
    """What the web app sees of a document."""

    id: str
    filename: str
    format: str
    width: int
    height: int
    revision: str
    """Identifies the current edit state; changes whenever the rendered result would."""
    state: EditState
    head: str | None
    tip: str | None
    can_undo: bool
    can_redo: bool
    undo_label: str | None
    """What undo would take back."""
    redo_label: str | None
    """What redo would bring back."""
    history: list[StepView]
    chat: list[ChatEntry]
    pending_plan: Plan | None = None
    """A plan the agent proposed, waiting for the go-ahead."""

    @classmethod
    def of(cls, doc: Document) -> DocumentView:
        active = {s.id for s in doc.lineage(doc.head)}
        current, following = doc.current, doc._next_toward_tip()
        return cls(
            id=doc.id,
            filename=doc.filename,
            format=doc.format,
            width=doc.width,
            height=doc.height,
            revision=doc.revision,
            state=doc.state,
            head=doc.head,
            tip=doc.tip,
            can_undo=doc.can_undo,
            can_redo=doc.can_redo,
            undo_label=current.label if current else None,
            redo_label=following.label if following else None,
            history=[
                StepView(
                    id=s.id,
                    parent=s.parent,
                    kind=s.kind,
                    label=s.label,
                    created_at=s.created_at,
                    active=s.id in active,
                )
                for s in doc.steps
            ],
            chat=doc.chat,
            pending_plan=doc.pending_plan,
        )
