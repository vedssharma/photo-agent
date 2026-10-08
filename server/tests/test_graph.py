import pytest
from pydantic import ValidationError

from photo_agent.graph import Document, DocumentView, Step
from photo_agent.operations import (
    OPERATIONS_BY_NAME,
    Crop,
    Exposure,
    Operation,
    OperationAdapter,
    Saturation,
)


def make_doc() -> Document:
    return Document(filename="a.jpg", format="JPEG", width=10, height=10)


def test_new_document_has_no_operations() -> None:
    doc = make_doc()
    assert doc.operations == []
    assert not doc.can_undo
    assert not doc.can_redo
    assert doc.revision == "original"


def step(label: str, *ops: Operation) -> Step:
    return Step(label=label, operations=list(ops))


def test_steps_snapshot_operations_and_undo_redo_walk_the_branch() -> None:
    doc = make_doc()
    first: list[Operation] = [Exposure(stops=0.5)]
    second: list[Operation] = [*first, Saturation(amount=20)]
    doc.commit(step("brighter", *first))
    doc.commit(step("more color", *second))
    assert doc.operations == second
    assert doc.steps[1].parent == doc.steps[0].id

    assert doc.undo()
    assert doc.operations == first
    assert doc.can_redo
    assert doc.undo()
    assert doc.operations == []
    assert not doc.undo()

    assert doc.redo()
    assert doc.operations == first
    assert doc.redo()
    assert doc.operations == second
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
    assert doc.operations == b.operations
    doc.undo()
    assert doc.redo()
    assert doc.head == b.id
    assert doc.chat[-1].text == "Redid: b"


def test_jump_to_original_and_back() -> None:
    doc = make_doc()
    doc.commit(step("a", Exposure(stops=1)))
    doc.commit(step("b", Exposure(stops=2)))
    assert doc.checkout(None)
    assert doc.operations == []
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
    doc.commit(step("same again", Exposure(id="e1", stops=1)))
    assert doc.revision == first
    doc.commit(step("b", Exposure(id="e1", stops=2)))
    assert doc.revision != first


def test_document_round_trips_through_json() -> None:
    doc = make_doc()
    doc.commit(step("crop", Crop(aspect="4:5"), Exposure(stops=0.3)))
    again = Document.model_validate_json(doc.model_dump_json())
    assert again == doc
    assert isinstance(again.operations[0], Crop)


def test_reads_phase_1_documents() -> None:
    v1 = {
        "id": "abcabcabcabc",
        "filename": "a.jpg",
        "format": "JPEG",
        "width": 10,
        "height": 10,
        "turns": [
            {"id": "t1", "request": "warmer", "reply": "ok", "operations": []},
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
