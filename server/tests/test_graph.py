import pytest
from pydantic import ValidationError

from photo_agent.graph import Document, DocumentView, Step
from photo_agent.layers import EditState
from photo_agent.operations import (
    OPERATIONS_BY_NAME,
    Crop,
    Exposure,
    OpBase,
    Operation,
    OperationAdapter,
    Saturation,
)


def make_doc() -> Document:
    return Document(filename="a.jpg", format="JPEG", width=10, height=10)


def test_new_document_has_no_operations() -> None:
    doc = make_doc()
    assert doc.state.is_empty
    assert not doc.can_undo
    assert not doc.can_redo
    assert doc.revision == "original"


def step(label: str, *ops: Operation) -> Step:
    return Step(label=label, state=EditState.from_operations(ops, label))


def ops_of(doc: Document) -> list[OpBase]:
    return list(doc.state.all_operations())


def test_steps_snapshot_operations_and_undo_redo_walk_the_branch() -> None:
    doc = make_doc()
    first: list[Operation] = [Exposure(stops=0.5)]
    second: list[Operation] = [*first, Saturation(amount=20)]
    doc.commit(step("brighter", *first))
    doc.commit(step("more color", *second))
    assert ops_of(doc) == second
    assert doc.steps[1].parent == doc.steps[0].id

    assert doc.undo()
    assert ops_of(doc) == first
    assert doc.can_redo
    assert doc.undo()
    assert ops_of(doc) == []
    assert not doc.undo()

    assert doc.redo()
    assert ops_of(doc) == first
    assert doc.redo()
    assert ops_of(doc) == second
    assert not doc.redo()
    assert [e.text for e in doc.chat] == [
        "Undid: more color",
        "Undid: brighter",
        "Redid: brighter",
        "Redid: more color",
    ]


def test_editing_after_undo_starts_a_branch_and_keeps_the_old_one() -> None:
    doc = make_doc()
    doc.commit(step("a", Exposure(stops=1)))
    doc.commit(step("b", Exposure(stops=2)))
    doc.undo()
    doc.commit(step("c", Exposure(stops=-1)))
    a, b, c = doc.steps
    assert [s.label for s in doc.steps] == ["a", "b", "c"]
    assert b.parent == c.parent == a.id
    assert not doc.can_redo

    # Jump back to the other branch; redo then follows that branch.
    assert doc.checkout(a.id)
    assert doc.redo()
    assert doc.head == c.id
    doc.checkout(b.id)
    assert doc.state == b.state
    doc.undo()
    assert doc.redo()
    assert doc.head == b.id
    assert doc.chat[-1].text == "Redid: b"


def test_jump_to_original_and_back() -> None:
    doc = make_doc()
    doc.commit(step("a", Exposure(stops=1)))
    doc.commit(step("b", Exposure(stops=2)))
    assert doc.checkout(None)
    assert ops_of(doc) == []
    assert doc.redo()
    assert doc.redo()
    assert doc.head == doc.steps[1].id
    assert not doc.checkout(doc.head)
    with pytest.raises(KeyError):
        doc.checkout("nope")


def test_revision_follows_the_rendered_result() -> None:
    doc = make_doc()
    doc.commit(step("a", Exposure(id="e1", stops=1)))
    first = doc.revision
    renamed = doc.state
    renamed.layers[0].name = "Renamed"
    doc.commit(Step(label="rename", state=renamed))
    assert doc.revision == first
    doc.commit(step("b", Exposure(id="e1", stops=2)))
    assert doc.revision != first


def test_document_round_trips_through_json() -> None:
    doc = make_doc()
    doc.commit(step("crop", Crop(aspect="4:5"), Exposure(stops=0.3)))
    again = Document.model_validate_json(doc.model_dump_json())
    assert again == doc
    assert isinstance(again.state.framing[0], Crop)


def test_reads_phase_1_documents() -> None:
    v1 = {
        "id": "abcabcabcabc",
        "filename": "a.jpg",
        "format": "JPEG",
        "width": 10,
        "height": 10,
        "turns": [
            {
                "id": "t1",
                "request": "warmer",
                "reply": "ok",
                "operations": [
                    {"op": "exposure", "id": "e", "stops": 1},
                    {"op": "crop", "id": "c", "aspect": "1:1"},
                    {"op": "vignette", "id": "v", "amount": -20},
                ],
            },
            {"id": "t2", "request": "brighter", "reply": "ok", "operations": []},
        ],
        "cursor": 1,
        "chat": [{"role": "assistant", "text": "ok", "turn_id": "t1"}],
    }
    doc = Document.model_validate(v1)
    assert [(s.id, s.parent, s.label) for s in doc.steps] == [
        ("t1", None, "warmer"),
        ("t2", "t1", "brighter"),
    ]
    assert (doc.head, doc.tip) == ("t1", "t2")
    assert doc.chat[0].step_id == "t1"
    state = doc.steps[0].state
    assert [op.id for op in state.framing] == ["c"]
    assert [lay.name for lay in state.layers] == ["warmer"]
    assert [op.id for op in state.layers[0].operations] == ["e", "v"]


def test_view_marks_steps_that_are_in_effect() -> None:
    doc = make_doc()
    doc.commit(step("a"))
    doc.commit(step("b"))
    doc.undo()
    view = DocumentView.of(doc)
    assert [s.active for s in view.history] == [True, False]
    assert (view.undo_label, view.redo_label) == ("a", "b")


def test_operations_validate_their_parameters() -> None:
    with pytest.raises(ValidationError):
        Exposure(stops=9)
    with pytest.raises(ValidationError):
        Crop(left=0.8, right=0.2)
    with pytest.raises(ValidationError):
        OperationAdapter.validate_python({"op": "exposure", "stops": 1, "bogus": True})
    op = OperationAdapter.validate_python({"op": "saturation", "amount": -100})
    assert isinstance(op, Saturation)


def test_every_operation_has_a_description_for_the_agent() -> None:
    for name, cls in OPERATIONS_BY_NAME.items():
        assert cls.__doc__, name
        assert cls.model_json_schema()["properties"]["op"]["const"] == name


def test_summary_is_readable() -> None:
    assert Exposure(stops=0.4).summary() == "Exposure (stops +0.4)"


def test_manual_tweaks_of_the_same_control_fold_into_one_step() -> None:
    doc = make_doc()
    doc.commit(step("warmer", Exposure(id="e", stops=0.5)))
    base = doc.state

    def with_stops(stops: float) -> EditState:
        state = base.model_copy(deep=True)
        state.layers[0].operations[0] = Exposure(id="e", stops=stops)
        return state

    assert doc.edit_by_hand("Exposure +0.6", with_stops(0.6), coalesce="e.stops")
    assert doc.edit_by_hand("Exposure +0.7", with_stops(0.7), coalesce="e.stops")
    assert [s.label for s in doc.steps] == ["warmer", "Exposure +0.7"]
    assert [e.text for e in doc.chat] == ["Edited by hand: Exposure +0.7"]
    assert not doc.edit_by_hand("Same", with_stops(0.7), coalesce="e.stops")

    assert doc.edit_by_hand("Exposure +0.2", with_stops(0.2), coalesce="other")
    assert len(doc.steps) == 3
    doc.undo()
    # The step now has a child, so a new tweak starts its own step.
    assert doc.edit_by_hand("Exposure +0.9", with_stops(0.9), coalesce="e.stops")
    assert len(doc.steps) == 4
