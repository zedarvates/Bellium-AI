"""ShapeGrammar 3D plan contract and deterministic fixture evidence."""

from copy import deepcopy

import pytest

from bellium.geometry import FIXTURE_NAMES, check_plan, fixture, plan_from_dict, plan_to_dict


def test_fixture_is_valid_and_roundtrips() -> None:
    plan = fixture("sci-fi-hammer")
    assert check_plan(plan) == []
    rebuilt = plan_from_dict(plan_to_dict(plan))
    assert rebuilt == plan


def test_fixture_is_deterministic_and_not_mutated() -> None:
    first = plan_to_dict(fixture("sci-fi-hammer"))
    snapshot = deepcopy(first)
    second = plan_to_dict(fixture("sci-fi-hammer"))
    assert first == snapshot == second
    assert FIXTURE_NAMES == ("sci-fi-hammer",)


def test_fixture_preserves_reversible_operation_provenance() -> None:
    plan = fixture("sci-fi-hammer")
    assert len(plan.operations) == 2
    assert all(operation.reversible for operation in plan.operations)
    assert all(operation.source_evidence == "fixture-rule" for operation in plan.operations)


def test_unknown_root_key_fails_closed() -> None:
    payload = plan_to_dict(fixture("sci-fi-hammer"))
    payload["magic"] = True
    with pytest.raises(ValueError, match="unknown construction plan keys"):
        plan_from_dict(payload)


def test_missing_operation_input_is_rejected() -> None:
    payload = plan_to_dict(fixture("sci-fi-hammer"))
    payload["operations"][0]["input_ids"] = ["does-not-exist"]
    with pytest.raises(ValueError, match="operation_0_missing_input"):
        plan_from_dict(payload)


def test_duplicate_output_is_rejected() -> None:
    payload = plan_to_dict(fixture("sci-fi-hammer"))
    payload["operations"][0]["output_ids"] = ["head"]
    with pytest.raises(ValueError, match="operation_0_duplicate_output"):
        plan_from_dict(payload)


def test_surface_pattern_must_target_known_object() -> None:
    payload = plan_to_dict(fixture("sci-fi-hammer"))
    payload["surface_patterns"][0]["target_ids"] = ["unknown"]
    with pytest.raises(ValueError, match="surface_pattern_0_missing_target"):
        plan_from_dict(payload)


def test_confidence_is_bounded() -> None:
    payload = plan_to_dict(fixture("sci-fi-hammer"))
    payload["operations"][0]["confidence"] = 1.1
    with pytest.raises(ValueError, match="between 0 and 1"):
        plan_from_dict(payload)


def test_contract_records_downstream_handoff_without_executing_geometry() -> None:
    plan = fixture("sci-fi-hammer")
    assert plan.handoff["run_approx_surface"] is True
    assert plan.handoff["run_topology_grammar"] is True
    assert plan.handoff["generate_uv"] is True
    assert plan.provenance["notes"] == "No geometry execution claim."
