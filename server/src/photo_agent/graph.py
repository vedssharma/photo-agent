"""Edit graph v0: a document is the original image plus an ordered list of operations.

The graph never stores pixels. Each chat turn with the agent records the full operation
list as it stood after that turn, so undo and redo are just moving a cursor over turns, and
any step can be re-rendered at preview or full resolution from the original.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from pydantic import BaseModel, Field

from photo_agent.operations import Operation


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def now() -> datetime:
    return datetime.now(UTC)


class Turn(BaseModel):
    """One exchange with the agent and the edit state it left behind."""

    id: str = Field(default_factory=new_id)
    request: str
    """What the user asked for."""
    reply: str = ""
    """The agent's plain-language explanation of what it changed."""
    operations: list[Operation] = Field(default_factory=list)
    """The complete operation list after this turn."""
    created_at: datetime = Field(default_factory=now)


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
    turns: list[Turn] = Field(default_factory=list)
    cursor: int = 0
    """How many turns are applied. Turns past the cursor were undone and can be redone."""

    @property
    def applied_turns(self) -> list[Turn]:
        return self.turns[: self.cursor]

    @property
    def operations(self) -> list[Operation]:
        """The operations currently in effect."""
        return list(self.turns[self.cursor - 1].operations) if self.cursor else []

    @property
    def revision(self) -> str:
        return self.turns[self.cursor - 1].id if self.cursor else "original"

    @property
    def can_undo(self) -> bool:
        return self.cursor > 0

    @property
    def can_redo(self) -> bool:
        return self.cursor < len(self.turns)

    def commit_turn(self, turn: Turn) -> None:
        """Apply a new turn. Anything previously undone is discarded, as in any editor."""
        del self.turns[self.cursor :]
        self.turns.append(turn)
        self.cursor = len(self.turns)

    def undo(self) -> bool:
        if not self.can_undo:
            return False
        self.cursor -= 1
        return True

    def redo(self) -> bool:
        if not self.can_redo:
            return False
        self.cursor += 1
        return True


class TurnView(BaseModel):
    id: str
    request: str
    reply: str
    operations: list[Operation]
    applied: bool


class DocumentView(BaseModel):
    """What the web app sees of a document."""

    id: str
    filename: str
    format: str
    width: int
    height: int
    step: int
    """Number of applied turns."""
    revision: str
    """Identifies the current edit state; changes whenever the rendered result would."""
    operations: list[Operation]
    can_undo: bool
    can_redo: bool
    turns: list[TurnView]

    @classmethod
    def of(cls, doc: Document) -> DocumentView:
        return cls(
            id=doc.id,
            filename=doc.filename,
            format=doc.format,
            width=doc.width,
            height=doc.height,
            step=doc.cursor,
            revision=doc.revision,
            operations=doc.operations,
            can_undo=doc.can_undo,
            can_redo=doc.can_redo,
            turns=[
                TurnView(
                    id=t.id,
                    request=t.request,
                    reply=t.reply,
                    operations=t.operations,
                    applied=i < doc.cursor,
                )
                for i, t in enumerate(doc.turns)
            ],
        )
