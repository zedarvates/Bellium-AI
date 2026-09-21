"""Bounded consequence prediction: deterministic veto, precedents, then a micro-NN.

The specialist estimates what a proposed action would do. It never runs it, and
an irreversible action without a backup or a preview never reaches a model.
"""

from __future__ import annotations

from typing import Any

from bellium.contracts.schema import AuthorityMode, SpecialistResult
from bellium.knn.consequence import (
    BASELINE_ID,
    LABELS,
    consequence_features,
    deterministic_risk_label,
    retrieve_precedent,
)
from bellium.micro_nn.features import featurize
from bellium.micro_nn.mlp import load_mlp, predict_mlp
from bellium.resources import model_path

SPECIALIST_ID = "bellium/hybrid/consequence-predictor:v0"
NN_ID = "bellium/micro-nn/consequence-predictor:v0"
MODEL_PATH = model_path("micro_nn", "consequence-predictor", "v0.json")
CONSTRAINTS = ("irreversible", "dry_run_available", "has_backup")
ABSTAIN_THRESHOLD = 0.55
FEATURE_SCHEMA = "consequence_predictor"


def _flag(constraints: dict[str, Any], name: str) -> bool | None:
    if name not in constraints:
        return None
    value = constraints[name]
    if not isinstance(value, bool):
        raise ValueError(f"constraint {name} must be a boolean")
    return value


def classify_consequence(state: dict[str, Any]) -> SpecialistResult:
    """Micro-NN only. Returns a consequence label, never a decision."""
    vector = featurize(FEATURE_SCHEMA, consequence_features(state))
    probs = predict_mlp(load_mlp(MODEL_PATH), vector)
    best = max(range(len(probs)), key=probs.__getitem__)
    confidence = float(probs[best])
    abstained = confidence < ABSTAIN_THRESHOLD
    return SpecialistResult(
        NN_ID,
        {
            "label": None if abstained else LABELS[best],
            "probabilities": {LABELS[i]: round(float(probs[i]), 4) for i in range(len(LABELS))},
        },
        round(confidence, 4),
        abstained,
        AuthorityMode.CONSULTATIVE,
        evidence_ref=MODEL_PATH.name,
        notes=("Compact consequence classifier. It imitates a documented risk rule.",),
    )


def predict_consequence(query: dict[str, Any]) -> SpecialistResult:
    state = query.get("state")
    if not isinstance(state, dict):
        raise ValueError("query needs a state object")
    constraints = query.get("constraints", {})
    if not isinstance(constraints, dict):
        raise ValueError("constraints must be an object")
    unknown = sorted(set(constraints) - set(CONSTRAINTS))
    if unknown:
        raise ValueError(f"unknown constraints refuse to be ignored: {', '.join(unknown)}")
    declared = {name: _flag(constraints, name) for name in CONSTRAINTS}
    features = consequence_features(state)
    reference = deterministic_risk_label(features)
    shared = {
        "baseline": reference, "baseline_id": BASELINE_ID,
        "requires_confirmation": True, "executes": False, "certified": False,
    }
    unsafe_declared = declared["irreversible"] is True and not (
        declared["dry_run_available"] is True or declared["has_backup"] is True
    )
    unsafe_state = features["reversible"] < 0.5 and features["has_backup"] < 0.5 and features["dry_run_available"] < 0.5
    if unsafe_declared or unsafe_state:
        return SpecialistResult(
            SPECIALIST_ID,
            {
                **shared,
                "status": "veto",
                "consequence": "dangerous",
                "reason": "irreversible_without_backup_or_preview",
                "baseline": reference,
                "baseline_id": BASELINE_ID,
                "requires_confirmation": True,
                "executes": False,
                "micro_nn": None,
                "precedent": None,
            },
            1.0, False, AuthorityMode.CONSULTATIVE,
            notes=("Deterministic veto returned before any model was consulted.",),
        )
    if any(value is None for value in declared.values()):
        return SpecialistResult(
            SPECIALIST_ID,
            {**shared, "status": "abstain", "consequence": None,
             "reason": "unknown_constraints", "micro_nn": None, "precedent": None},
            0.0, True, AuthorityMode.CONSULTATIVE,
        )
    consistent = (
        declared["irreversible"] == (features["reversible"] < 0.5)
        and declared["has_backup"] == (features["has_backup"] >= 0.5)
        and declared["dry_run_available"] == (features["dry_run_available"] >= 0.5)
    )
    if not consistent:
        return SpecialistResult(
            SPECIALIST_ID,
            {**shared, "status": "abstain", "consequence": None,
             "reason": "conflicting_safety_state", "micro_nn": None, "precedent": None},
            0.0, True, AuthorityMode.CONSULTATIVE,
        )
    precedent = retrieve_precedent({"family": query.get("family"), "features": features})
    if precedent.abstained:
        return SpecialistResult(
            SPECIALIST_ID,
            {**shared, "status": "abstain", "consequence": None,
             "reason": "no_verified_precedent", "micro_nn": None, "precedent": precedent.output},
            0.0, True, AuthorityMode.CONSULTATIVE,
            notes=("Without verified precedents the estimate fails closed before the micro-NN.",),
        )
    proposal = classify_consequence(state)
    if proposal.abstained:
        return SpecialistResult(
            SPECIALIST_ID,
            {**shared, "status": "abstain", "consequence": None,
             "reason": "micro_nn_abstained",
             "micro_nn": proposal.output, "precedent": precedent.output},
            0.0, True, AuthorityMode.CONSULTATIVE,
            notes=("No consequence is proposed when the classifier is unsure.",),
        )
    label = proposal.output["label"]
    if label != reference or precedent.output["label"] != reference:
        return SpecialistResult(
            SPECIALIST_ID,
            {**shared, "status": "abstain", "consequence": None,
             "reason": "learned_tier_differs_from_rule",
             "agrees_with_baseline": False,
             "micro_nn": proposal.output, "precedent": precedent.output},
            0.0, True, AuthorityMode.CONSULTATIVE,
            notes=(
                "The published rule stays the reference; a learned tier that "
                "contradicts it produces an abstention, not a promotion.",
            ),
        )
    return SpecialistResult(
        SPECIALIST_ID,
        {
            **shared,
            "status": "suggest",
            "consequence": label,
            "certified": False,
            "agrees_with_baseline": True,
            "micro_nn": proposal.output,
            "precedent": precedent.output,
        },
        min(proposal.confidence, precedent.confidence or 0.0),
        False, AuthorityMode.CONSULTATIVE,
        notes=(
            "Estimate only. Deterministic policy and the caller own execution.",
            "Three independent signals agree: the rule, verified precedents and the micro-NN.",
        ),
    )
