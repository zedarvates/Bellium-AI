"""Advisory consequence estimates must not acquire authority through coercion."""

from copy import deepcopy
import json

import pytest

from bellium.contracts.schema import AuthorityMode
from bellium.knn.consequence import SCHEMA, load_precedents, retrieve_precedent
from bellium.specialists import consequence as specialist
from bellium.specialists.consequence import classify_consequence


@pytest.fixture
def query():
    return {
        "family": "filesystem",
        "state": {
            "reversible": 1.0, "scope_ratio": 0.15, "touches_protected": 0.0,
            "has_backup": 1.0, "dry_run_available": 1.0, "blast_radius": 0.1,
            "idempotent": 1.0, "recent_failures": 0.05,
        },
        "constraints": {
            "irreversible": False, "has_backup": True, "dry_run_available": True,
        },
    }


@pytest.fixture
def no_learned_calls(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("invalid input must be stopped before consulting learned tiers")

    monkeypatch.setattr(specialist, "retrieve_precedent", unexpected)
    monkeypatch.setattr(specialist, "classify_consequence", unexpected)
    monkeypatch.setattr(specialist, "load_mlp", unexpected)


def _memory(query):
    return [
        {"id": f"case_{i}", "family": query["family"], "label": "safe",
         "verified": True, "source": "fixture:boundary-test",
         "features": dict(query["state"])}
        for i in range(3)
    ]


@pytest.mark.parametrize("constraints", [None, False, 0, "", []])
def test_falsey_non_objects_are_rejected(query, no_learned_calls, constraints):
    query["constraints"] = constraints
    with pytest.raises(ValueError, match="constraints must be an object"):
        specialist.predict_consequence(query)


@pytest.mark.parametrize("name", ["irreversible", "has_backup", "dry_run_available"])
@pytest.mark.parametrize("value", [None, "false", 0, 1])
def test_constraint_values_require_booleans(query, no_learned_calls, name, value):
    query["constraints"][name] = value
    with pytest.raises(ValueError, match="must be a boolean"):
        specialist.predict_consequence(query)


@pytest.mark.parametrize("name", ["irreversible", "has_backup", "dry_run_available", "all"])
def test_missing_constraints_abstain_before_models(query, no_learned_calls, name):
    if name == "all":
        query.pop("constraints")
    else:
        query["constraints"].pop(name)
    result = specialist.predict_consequence(query)
    assert result.abstained
    assert result.output["reason"] == "unknown_constraints"
    assert result.output["requires_confirmation"] is True
    assert result.output["executes"] is False
    assert result.output["certified"] is False


@pytest.mark.parametrize("name", ["reversible", "has_backup", "dry_run_available"])
def test_contradictory_safety_inputs_abstain(query, no_learned_calls, name):
    query["state"][name] = 0.0
    result = specialist.predict_consequence(query)
    assert result.abstained
    assert result.output["reason"] == "conflicting_safety_state"


def test_state_cannot_hide_irreversibility_by_omitting_constraints(query, no_learned_calls):
    query["state"].update(reversible=0.0, has_backup=0.0, dry_run_available=0.0)
    query.pop("constraints")
    result = specialist.predict_consequence(query)
    assert result.output["status"] == "veto"
    assert result.output["micro_nn"] is None
    assert result.output["precedent"] is None


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf"), None, True])
@pytest.mark.parametrize("tier", ["knn", "micro", "hybrid"])
def test_invalid_features_are_not_clamped_or_coerced(query, no_learned_calls, value, tier):
    query["state"]["blast_radius"] = value
    with pytest.raises(ValueError):
        if tier == "knn":
            retrieve_precedent({"family": query["family"], "features": query["state"]})
        elif tier == "micro":
            classify_consequence(query["state"])
        else:
            specialist.predict_consequence(query)


@pytest.mark.parametrize("verified", ["false", "true", 0, 1, None, [], {}])
@pytest.mark.parametrize("source", ["disk", "injected"])
def test_verified_is_not_truthiness(query, tmp_path, verified, source):
    records = _memory(query)
    for item in records:
        item["verified"] = verified
    with pytest.raises(ValueError, match="verified must be a boolean"):
        if source == "disk":
            path = tmp_path / "precedents.json"
            path.write_text(json.dumps({"schema": SCHEMA, "items": records}), encoding="utf-8")
            load_precedents(path)
        else:
            retrieve_precedent({"family": query["family"], "features": query["state"]},
                               memory=records)


def test_duplicate_ids_cannot_supply_three_precedents(query):
    records = _memory(query)
    with pytest.raises(ValueError, match="duplicate precedent id"):
        retrieve_precedent({"family": query["family"], "features": query["state"]},
                           memory=[records[0]] * 3)


def test_safe_label_still_requires_caller_confirmation_and_preserves_input(query):
    before = deepcopy(query)
    result = specialist.predict_consequence(query)
    assert result.output["status"] == "suggest"
    assert result.output["consequence"] == "safe"
    assert result.output["requires_confirmation"] is True
    assert result.output["executes"] is False
    assert result.output["certified"] is False
    assert result.authority_mode is AuthorityMode.CONSULTATIVE
    assert query == before


def test_absent_family_support_skips_neural_model(query, monkeypatch):
    query["family"] = "unknown"

    def unexpected(*args, **kwargs):
        pytest.fail("no neural prediction is needed without precedent support")

    monkeypatch.setattr(specialist, "load_mlp", unexpected)
    result = specialist.predict_consequence(query)
    assert result.abstained
    assert result.output["reason"] == "no_verified_precedent"
    assert result.output["micro_nn"] is None
