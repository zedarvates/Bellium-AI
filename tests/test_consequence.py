from bellium.contracts.schema import AuthorityMode, SpecialistResult
from bellium.knn.consequence import (
    LABELS,
    load_precedents,
    retrieve_precedent,
)
from bellium.specialists import consequence as consequence_specialist
from bellium.specialists.consequence import classify_consequence, predict_consequence

SAFE = {
    "reversible": 1.0,
    "scope_ratio": 0.15,
    "touches_protected": 0.0,
    "has_backup": 1.0,
    "dry_run_available": 1.0,
    "blast_radius": 0.10,
    "idempotent": 1.0,
    "recent_failures": 0.05,
}
SAFE_CONSTRAINTS = {"irreversible": False, "has_backup": True, "dry_run_available": True}
REVIEW = {
    "reversible": 1.0,
    "scope_ratio": 0.70,
    "touches_protected": 0.0,
    "has_backup": 1.0,
    "dry_run_available": 0.0,
    "blast_radius": 0.70,
    "idempotent": 0.30,
    "recent_failures": 0.50,
}
DANGEROUS = {
    "reversible": 0.0,
    "scope_ratio": 0.90,
    "touches_protected": 1.0,
    "has_backup": 0.0,
    "dry_run_available": 0.0,
    "blast_radius": 0.90,
    "idempotent": 0.0,
    "recent_failures": 0.60,
}


def test_memory_is_labelled_by_the_published_rule() -> None:
    precedents = load_precedents()
    assert precedents
    assert all(item["label"] in LABELS for item in precedents)
    assert any(item["verified"] for item in precedents)


def test_irreversible_action_without_backup_is_vetoed() -> None:
    result = predict_consequence(
        {"family": "deployment", "state": REVIEW, "constraints": {"irreversible": True}}
    )
    assert result.output["status"] == "veto"
    assert result.output["consequence"] == "dangerous"
    assert result.output["reason"] == "irreversible_without_backup_or_preview"
    assert result.output["micro_nn"] is None
    assert result.output["executes"] is False


def test_dry_run_makes_the_veto_unnecessary() -> None:
    result = predict_consequence(
        {
            "family": "deployment",
            "state": REVIEW,
            "constraints": {"irreversible": True, "dry_run_available": True},
        }
    )
    assert result.output["status"] != "veto"


def test_safe_action_is_proposed_with_every_signal_agreeing() -> None:
    # Explicit constraints are required by docs/CONSEQUENCE_CONTRACT.md.
    result = predict_consequence({"family": "filesystem", "state": SAFE, "constraints": SAFE_CONSTRAINTS})
    assert result.output["status"] == "suggest"
    assert result.output["consequence"] == "safe"
    assert result.output["agrees_with_baseline"] is True
    assert result.output["requires_confirmation"] is True
    assert result.output["executes"] is False


def test_precedent_must_agree_with_the_rule(monkeypatch) -> None:
    def conflicting(query, *, memory=None, k=5):
        return SpecialistResult(
            "bellium/knn/consequence-precedent:v0",
            {"status": "suggest", "label": "dangerous", "neighbors": [], "executes": False},
            0.9, False, AuthorityMode.CONSULTATIVE,
        )

    monkeypatch.setattr(consequence_specialist, "retrieve_precedent", conflicting)
    result = predict_consequence({"family": "filesystem", "state": SAFE, "constraints": SAFE_CONSTRAINTS})
    assert result.abstained is True
    assert result.output["reason"] == "learned_tier_differs_from_rule"
    assert result.output["agrees_with_baseline"] is False


def test_unverified_precedents_do_not_decide() -> None:
    memory = [
        {
            "id": f"unverified_{index}",
            "family": "filesystem",
            "label": "safe",
            "verified": False,
            "source": "fixture:test",
            "features": dict(SAFE),
        }
        for index in range(5)
    ]
    result = retrieve_precedent({"family": "filesystem", "features": SAFE}, memory=memory)
    assert result.abstained is True
    assert result.output["reason"] == "too_few_verified_precedents"


def test_micro_nn_alone_never_executes() -> None:
    result = classify_consequence(DANGEROUS)
    assert result.output["label"] in LABELS
    assert "executes" not in result.output


def test_unknown_constraint_is_refused() -> None:
    blocked = False
    try:
        predict_consequence(
            {"family": "filesystem", "state": SAFE, "constraints": {"invisible": True}}
        )
    except ValueError:
        blocked = True
    assert blocked


def test_missing_feature_is_not_coerced() -> None:
    partial = dict(SAFE)
    partial.pop("idempotent")
    blocked = False
    try:
        predict_consequence({"family": "filesystem", "state": partial, "constraints": {}})
    except ValueError:
        blocked = True
    assert blocked
