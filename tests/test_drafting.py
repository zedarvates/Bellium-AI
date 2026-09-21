"""2D drafting document: invariants and SVG/DXF round-trips on authored fixtures."""

from copy import deepcopy

import pytest

from bellium.contracts.schema import AuthorityMode
from bellium.drafting.document import (
    UnsupportedDrafting,
    check_drawing,
    drawing_from_dict,
    drawing_to_dict,
    drawings_match,
)
from bellium.drafting.dxf import drawing_from_dxf, drawing_to_dxf
from bellium.drafting.fixtures import FIXTURE_NAMES, fixture
from bellium.drafting.svg import drawing_from_svg, drawing_to_svg
from bellium.registry.catalog import get_specialist
from bellium.specialists.drafting import inspect_drawing


def test_unknown_keys_raise() -> None:
    payload = drawing_to_dict(fixture("washer"))
    payload["secret"] = True
    with pytest.raises(ValueError, match="unknown drawing keys"):
        drawing_from_dict(payload)


def test_zero_radius_raises() -> None:
    payload = drawing_to_dict(fixture("washer"))
    payload["entities"][1]["r"] = 0
    with pytest.raises(ValueError, match="r must be > 0"):
        drawing_from_dict(payload)


def test_missing_layer_is_a_violation() -> None:
    drawing = fixture("washer")
    payload = drawing_to_dict(drawing)
    payload["entities"][0]["layer"] = "missing"
    rebuilt = drawing_from_dict(payload)
    assert "entity_0_missing_layer" in check_drawing(rebuilt)


def test_unsupported_kind_is_not_approximated() -> None:
    payload = drawing_to_dict(fixture("washer"))
    payload["entities"].append({"kind": "spline", "layer": "outline"})
    with pytest.raises(UnsupportedDrafting, match="unsupported_entity"):
        drawing_from_dict(payload)


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_svg_roundtrip_keeps_the_document(name: str) -> None:
    drawing = fixture(name)
    svg = drawing_to_svg(drawing)
    back = drawing_from_svg(svg)
    assert drawings_match(drawing, back)
    assert 'data-bellium="drafting-v0"' in svg
    assert 'data-y="up"' in svg


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_dxf_roundtrip_keeps_the_document(name: str) -> None:
    drawing = fixture(name)
    dxf = drawing_to_dxf(drawing)
    back = drawing_from_dxf(dxf)
    assert drawings_match(drawing, back)
    assert drawing.unit == back.unit
    assert "$INSUNITS" in dxf


def test_svg_keeps_y_up_instead_of_flipping_model_space() -> None:
    drawing = fixture("washer")
    back = drawing_from_svg(drawing_to_svg(drawing))
    hole = back.entities[1]
    assert hole.cy == 50.0
    assert hole.cx == 50.0


def test_inch_unit_survives_both_formats() -> None:
    drawing = fixture("inch-card")
    assert drawing.unit == "in"
    assert drawing_from_svg(drawing_to_svg(drawing)).unit == "in"
    assert drawing_from_dxf(drawing_to_dxf(drawing)).unit == "in"


def test_foreign_svg_abstains() -> None:
    result = inspect_drawing({"svg": "<svg xmlns='http://www.w3.org/2000/svg'></svg>"})
    assert result.abstained
    assert result.output["reason"] == "unsupported_svg_profile"


def test_unsupported_dxf_entity_abstains() -> None:
    dxf = drawing_to_dxf(fixture("washer"))
    injected = dxf.replace("EOF", "SPLINE" + chr(10) + "8" + chr(10) + "outline" + chr(10) + "0" + chr(10) + "EOF")
    result = inspect_drawing({"dxf": injected})
    assert result.abstained
    assert result.output["reason"] == "unsupported_dxf_entity"
    assert "SPLINE" in result.output["details"]


def test_empty_drawing_abstains() -> None:
    payload = drawing_to_dict(fixture("washer"))
    payload["entities"] = []
    result = inspect_drawing({"drawing": payload})
    assert result.abstained
    assert result.output["reason"] == "empty_drawing"


def test_specialist_emits_verified_svg_and_dxf() -> None:
    drawing = fixture("slot")
    result = inspect_drawing({"drawing": drawing_to_dict(drawing)})
    assert result.abstained is False
    assert result.authority_mode is AuthorityMode.CONSULTATIVE
    assert result.output["status"] == "ready"
    assert result.output["svg_roundtrip"] is True
    assert result.output["dxf_roundtrip"] is True
    assert result.output["pixels_written"] is False
    assert result.output["certified"] is False
    assert get_specialist(result.specialist_id).decision_eligible is False
    assert drawings_match(drawing, drawing_from_svg(result.output["svg"]))
    assert drawings_match(drawing, drawing_from_dxf(result.output["dxf"]))


def test_specialist_refuses_unknown_query_keys() -> None:
    with pytest.raises(ValueError, match="unknown query keys"):
        inspect_drawing({"drawing": drawing_to_dict(fixture("washer")), "render": True})


def test_fixtures_are_not_mutated_by_export() -> None:
    original = drawing_to_dict(fixture("bracket"))
    snapshot = deepcopy(original)
    inspect_drawing({"drawing": original})
    assert original == snapshot
