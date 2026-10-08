import pytest
from pydantic import ValidationError

from photo_agent.graph import Document, DocumentView, Turn
from photo_agent.operations import (
    OPERATIONS_BY_NAME,
    Crop,
    Exposure,
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


def test_turns_snapshot_operations_and_undo_redo_moves_the_cursor() -> None:
    doc = make_doc()
    first = [Exposure(stops=0.5)]
    second = [*first, Saturation(amount=20)]
    doc.commit_turn(Turn(request="brighter", operations=first))
    doc.commit_turn(Turn(request="more color", operations=second))
    assert doc.operations == second

    assert doc.undo()
    assert doc.operations == first
    assert doc.can_redo
    assert doc.undo()
    assert doc.operations == []
    assert not doc.undo()

    assert doc.redo()
    assert doc.operations == first


def test_new_turn_after_undo_discards_the_redo_branch() -> None:
    doc = make_doc()
    doc.commit_turn(Turn(request="a", operations=[Exposure(stops=1)]))
    doc.commit_turn(Turn(request="b", operations=[Exposure(stops=2)]))
    doc.undo()
    doc.commit_turn(Turn(request="c", operations=[Exposure(stops=-1)]))
    assert [t.request for t in doc.turns] == ["a", "c"]
    assert not doc.can_redo


def test_document_round_trips_through_json() -> None:
    doc = make_doc()
    doc.commit_turn(Turn(request="crop", operations=[Crop(aspect="4:5"), Exposure(stops=0.3)]))
    again = Document.model_validate_json(doc.model_dump_json())
    assert again == doc
    assert isinstance(again.operations[0], Crop)


def test_view_marks_undone_turns() -> None:
    doc = make_doc()
    doc.commit_turn(Turn(request="a"))
    doc.commit_turn(Turn(request="b"))
    doc.undo()
    view = DocumentView.of(doc)
    assert [t.applied for t in view.turns] == [True, False]
    assert view.revision == doc.turns[0].id


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
