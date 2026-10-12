"""Batch editing: one look for many photos, with each photo evened out to match the set.

The look comes from one of three places: an instruction to the agent ("warm and bright,
like a summer afternoon"), which it carries out on a reference photo first; a saved recipe;
or the reference photo's own edits ("make these look like this one"). The look's portable
layers (see `recipes.portable`) then go on every other photo, replacing the looks those
photos had but keeping their local work (a removed sign, a brushed-in mask).

The same look on different photos still gives different results: one was shot in shade,
another in sun. So each photo also gets a "Match the set" layer that moves its color and
tone part of the way toward the reference as edited. That per-photo adjustment is what makes
"make all of these consistent" work.

Every change is one step in each photo's history, so it can be undone photo by photo. A
batch runs in the background; its status says how far it has got.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field, model_validator

from photo_agent.agent import AgentEvent, AgentService
from photo_agent.layers import EditState, Layer, new_layer_id
from photo_agent.operations import MatchReference, ReferenceStats, new_op_id
from photo_agent.recipes import Recipe, is_portable, portable
from photo_agent.render import reference_stats

if TYPE_CHECKING:
    from photo_agent.library.core import Library

log = logging.getLogger(__name__)

MATCH_LAYER = "Match the set"


class BatchRequest(BaseModel):
    ids: list[str] = Field(min_length=1, max_length=500, description="The photos to edit.")
    instruction: str | None = Field(
        None, max_length=2000, description="What the look should be, in words, for the agent."
    )
    recipe: str | None = Field(None, description="A saved recipe to apply instead.")
    reference: str | None = Field(
        None,
        description="The photo whose look the others follow (the first by default). With an "
        "instruction, the agent edits it first; with neither instruction nor recipe, its "
        "current edits are the look.",
    )
    match: int = Field(
        60,
        ge=0,
        le=100,
        description="How far to even out each photo's color and tone toward the reference; "
        "0 applies the look alone.",
    )

    @model_validator(mode="after")
    def _one_source(self) -> BatchRequest:
        if self.instruction and self.recipe:
            raise ValueError("give an instruction or a recipe, not both")
        if self.reference is not None and self.reference not in self.ids:
            raise ValueError("the reference must be one of the photos")
        return self


class BatchFailure(BaseModel):
    photo: str
    reason: str


class BatchStatus(BaseModel):
    id: str
    state: Literal["running", "done", "failed"] = "running"
    total: int
    done: int = 0
    failed: list[BatchFailure] = Field(default_factory=list)
    current: str | None = Field(None, description="What is happening now.")
    reference: str
    message: str | None = Field(
        None, description="The agent's reply about the look, or why the batch failed."
    )


def keep_local(state: EditState) -> list[Layer]:
    """The layers that belong to this photo alone: removals, generated content, and
    adjustments limited to something painted or picked out in it."""
    return [
        layer
        for layer in state.layers
        if layer.is_content or (layer.mask is not None and not is_portable(layer.mask))
    ]


def fresh(layers: list[Layer]) -> list[Layer]:
    """Copies with new ids, so each photo's layers are its own."""
    out = []
    for layer in layers:
        copy = layer.model_copy(deep=True, update={"id": new_layer_id()})
        for op in copy.operations:
            op.id = new_op_id()
        out.append(copy)
    return out


def with_look(state: EditState, look: list[Layer]) -> EditState:
    return state.model_copy(update={"layers": [*keep_local(state), *fresh(look)]}, deep=True)


def _look_of(state: EditState) -> list[Layer]:
    """A photo's look: its portable layers, less any evening-out from an earlier batch."""
    return [layer for layer in portable(state.layers) if layer.name != MATCH_LAYER]


def start(library: Library, request: BatchRequest) -> BatchStatus:
    status = BatchStatus(
        id=uuid.uuid4().hex[:12],
        total=len(request.ids),
        reference=request.reference or request.ids[0],
    )
    library.batches[status.id] = status
    return status


async def run(
    library: Library,
    request: BatchRequest,
    status: BatchStatus,
    agent: AgentService | None,
    recipe: Recipe | None,
) -> None:
    try:
        await _run(library, request, status, agent, recipe)
        status.state = "done"
    except Exception as exc:  # Reported in the status the web app polls.
        log.exception("Batch %s failed", status.id)
        status.state, status.message = "failed", str(exc) or type(exc).__name__
    status.current = None


async def _run(
    library: Library,
    request: BatchRequest,
    status: BatchStatus,
    agent: AgentService | None,
    recipe: Recipe | None,
) -> None:
    store = library.store
    ref_id = status.reference
    label = "Batch: " + (
        request.instruction or (f"recipe “{recipe.name}”" if recipe else "match the look")
    )
    label = label[:80]
    if request.instruction:
        if agent is None:
            raise ValueError("Editing by instruction needs a Claude API key.")
        status.current = f"Working out the look on {library.get(ref_id).filename}"

        async def quiet(event: AgentEvent) -> None:
            pass

        doc = await agent.run_turn(ref_id, request.instruction, quiet)
        status.message = doc.chat[-1].text if doc.chat else None
        look = _look_of(doc.state)
    elif recipe is not None:
        look = recipe.layers
    else:
        look = _look_of(store.get(ref_id).state)

    # The reference as it will look: what every other photo is evened out toward.
    reference_doc = store.get(ref_id)
    reference_state = (
        with_look(reference_doc.state, look) if recipe is not None else reference_doc.state
    )
    target = None
    if request.match:
        loaded = store.image(ref_id)
        rendered = await asyncio.to_thread(
            library.renders.get_or_render,
            ref_id,
            loaded.proxy,
            reference_state,
            loaded.proxy_context,
        )
        target = reference_stats(rendered)

    for photo_id in request.ids:
        try:
            name = library.get(photo_id).filename
            status.current = f"Editing {name}"
            await asyncio.to_thread(
                _edit_one, library, photo_id, ref_id, look, target, request.match, label, recipe
            )
        except Exception as exc:
            log.warning("Batch %s: %s failed: %s", status.id, photo_id, exc)
            status.failed.append(BatchFailure(photo=photo_id, reason=str(exc) or "Failed"))
        status.done += 1


def _edit_one(
    library: Library,
    photo_id: str,
    ref_id: str,
    look: list[Layer],
    target: ReferenceStats | None,
    match: int,
    label: str,
    recipe: Recipe | None,
) -> None:
    store = library.store
    doc = store.get(photo_id)
    if photo_id == ref_id:
        # The reference already has the look, unless it comes from a recipe.
        if recipe is None:
            return
        state = with_look(doc.state, look)
    else:
        state = with_look(doc.state, look)
        if target is not None and match:
            state.layers.append(
                Layer(
                    id=new_layer_id(),
                    name=MATCH_LAYER,
                    operations=[
                        MatchReference(
                            id=new_op_id(),
                            reference=f"set-{ref_id}"[:40],
                            amount=match,
                            stats=target,
                        )
                    ],
                )
            )
    if doc.edit_by_hand(label, state):
        store.save(doc)
