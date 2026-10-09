from fastapi.testclient import TestClient

from photo_agent.controls import operation_specs
from photo_agent.operations import OPERATIONS_BY_NAME, OperationAdapter


def test_every_operation_has_controls() -> None:
    specs = {s.op: s for s in operation_specs()}
    assert set(specs) == set(OPERATIONS_BY_NAME)
    (stops,) = specs["exposure"].params
    assert (stops.kind, stops.min, stops.max, stops.default) == ("number", -5, 5, 0)
    assert specs["crop"].framing and not specs["exposure"].framing
    band = next(p for p in specs["hsl"].params if p.name == "band")
    assert band.kind == "choice" and band.choices and "blue" in band.choices
    generate = specs["generate"]
    assert [p.kind for p in generate.params] == ["text", "seed", "number"]
    assert generate.group == "generative"


def test_defaults_make_valid_operations() -> None:
    for spec in operation_specs():
        if not spec.addable:
            continue
        # Prompts are typed by the person; the rest have usable defaults.
        args = {p.name: "a plant" if p.kind == "text" else p.default for p in spec.params}
        OperationAdapter.validate_python({"op": spec.op, **args})


def test_operations_endpoint(client: TestClient) -> None:
    res = client.get("/api/operations")
    assert res.status_code == 200
    assert {s["op"] for s in res.json()} == set(OPERATIONS_BY_NAME)
