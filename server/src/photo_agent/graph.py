"""Edit graph: a document is the original image plus a tree of edit steps.

The graph never stores pixels. Each step (an agent turn or a manual tweak) records the full
operation list as it stood after that step, and points at the step it was made from. Undo
and redo move `head` along a branch; jumping to an older step and editing from there starts
a new branch without losing the old one. Any step can be re-rendered at preview or full
resolution from the original.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from photo_agent.operations import Operation


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
    operations: list[Operation] = Field(default_factory=list)
    """The complete operation list after this step."""
    created_at: datetime = Field(default_factory=now)


class ChatEntry(BaseModel):
    """One line of the conversation shown in the chat panel and replayed to the agent.

    The conversation is linear even when the history branches: undo, redo, and jumps are
    recorded as `event` entries, so the agent knows its earlier change is no longer in effect.
    """

    role: Literal["user", "assistant", "event"]
    text: str
    step_id: str | None = None
    """The step this entry produced, for assistant replies that changed the photo."""
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
    def operations(self) -> list[Operation]:
        """The operations currently in effect."""
        current = self.current
        return list(current.operations) if current else []

    @property
    def revision(self) -> str:
        """Identifies what the current edits render to; changes whenever the result would."""
        if self.head is None:
            return "original"
        payload = "[" + ",".join(op.model_dump_json() for op in self.operations) + "]"
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

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
    operations: list[Operation]
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
            operations=doc.operations,
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
        )
