"""Bounded NPC behaviour proposal: deterministic veto, then a micro-NN.

The classifier consumes caller-declared compact state. It never plans a path,
never executes an action, never learns online and never overrides a gameplay
rule: protected or forbidden situations are decided before the model runs.
"""

from __future__ import annotations

from typing import Any

from bellium.contracts.schema import AuthorityMode, SpecialistResult
from bellium.micro_nn.features import feature_names, featurize
from bellium.micro_nn.mlp import load_mlp, predict_mlp
from bellium.resources import model_path

SPECIALIST_ID = "bellium/hybrid/npc-behavior-router:v0"
NN_ID = "bellium/micro-nn/npc-behavior-router:v0"
MODEL_PATH = model_path("micro_nn", "npc-behavior-router", "v0.json")
LABELS = ("idle", "patrol", "investigate", "engage", "retreat")
ENGAGE = "engage"
HEALTH_INDEX = feature_names("npc_behavior_router").index("health_ratio")
ABSTAIN_THRESHOLD = 0.55
CONSTRAINTS = (
    "noncombatant",
    "surrendered",
    "incapacitated",
    "friendly_fire_risk",
    "forbidden_zone",
    "rules_of_engagement",
)
ENGAGEMENT_RULES = ("weapons_free", "defensive", "hold_fire")


def _flag(constraints: dict[str, Any], name: str) -> bool:
    value = constraints.get(name)
    if value is None:
        return False
    if not isinstance(value, bool):
        raise ValueError(f"constraint {name} must be a boolean")
    return value


def classify_npc_behavior(state: dict[str, Any]) -> SpecialistResult:
    """Micro-NN only. Returns a behaviour label, never an action."""
    vector = featurize("npc_behavior_router", state)
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
        notes=("Compact state classifier. It imitates a documented rule, not a learned policy.",),
    )


def route_npc_behavior(query: dict[str, Any]) -> SpecialistResult:
    state = query.get("state")
    if not isinstance(state, dict):
        raise ValueError("query needs a state object")
    constraints = query.get("constraints") or {}
    if not isinstance(constraints, dict):
        raise ValueError("constraints must be an object")
    unknown = sorted(set(constraints) - set(CONSTRAINTS))
    if unknown:
        raise ValueError(f"unknown constraints refuse to be ignored: {', '.join(unknown)}")
    rules = constraints.get("rules_of_engagement", "defensive")
    if rules not in ENGAGEMENT_RULES:
        raise ValueError(f"rules_of_engagement must be one of: {', '.join(ENGAGEMENT_RULES)}")
    declared = {name: _flag(constraints, name) for name in CONSTRAINTS if name != "rules_of_engagement"}
    vector = featurize("npc_behavior_router", state)
    health = float(vector[HEALTH_INDEX])
    if health <= 0.0 or declared["incapacitated"]:
        return SpecialistResult(
            SPECIALIST_ID,
            {"status": "veto", "behavior": "down", "reason": "incapacitated",
             "blocked_behaviors": list(LABELS), "executes": False, "micro_nn": None},
            1.0, False, AuthorityMode.CONSULTATIVE,
            notes=("Deterministic veto returned before any model was consulted.",),
        )
    protected = declared["noncombatant"] or declared["surrendered"]
    if protected:
        return SpecialistResult(
            SPECIALIST_ID,
            {"status": "veto", "behavior": "hold_fire", "reason": "protected_agent",
             "blocked_behaviors": [ENGAGE], "executes": False, "micro_nn": None},
            1.0, False, AuthorityMode.CONSULTATIVE,
            notes=("Protected agents never reach the classifier.",),
        )
    blocked = [ENGAGE] if (
        rules != "weapons_free"
        or declared["friendly_fire_risk"]
        or declared["forbidden_zone"]
    ) else []
    proposal = classify_npc_behavior(state)
    if proposal.abstained:
        return SpecialistResult(
            SPECIALIST_ID,
            {"status": "abstain", "behavior": None, "reason": "micro_nn_abstained",
             "blocked_behaviors": blocked, "executes": False, "micro_nn": proposal.output},
            0.0, True, AuthorityMode.CONSULTATIVE,
            notes=("No behaviour is proposed when the classifier is unsure.",),
        )
    label = proposal.output["label"]
    if label in blocked:
        return SpecialistResult(
            SPECIALIST_ID,
            {"status": "abstain", "behavior": None, "reason": "blocked_by_constraints",
             "blocked_behaviors": blocked, "executes": False, "micro_nn": proposal.output},
            0.0, True, AuthorityMode.CONSULTATIVE,
            notes=("The proposal contradicted a declared constraint; nothing is emitted.",),
        )
    return SpecialistResult(
        SPECIALIST_ID,
        {"status": "propose", "behavior": label, "reason": "classifier_proposal",
         "blocked_behaviors": blocked, "executes": False, "micro_nn": proposal.output,
         "certified": False},
        proposal.confidence,
        False,
        AuthorityMode.CONSULTATIVE,
        notes=("Proposal only. Gameplay code owns movement, combat and networking.",),
    )
