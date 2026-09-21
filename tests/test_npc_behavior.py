from bellium.micro_nn.features import feature_names
from bellium.micro_nn.mlp import load_mlp
from bellium.specialists.npc_behavior import (
    LABELS,
    MODEL_PATH,
    classify_npc_behavior,
    route_npc_behavior,
)

ENGAGING = {
    "threat_level": 0.9,
    "distance_ratio": 0.2,
    "health_ratio": 0.8,
    "support_ratio": 0.7,
    "visibility": 0.9,
    "alertness": 0.8,
    "cover_ratio": 0.6,
    "cooldown_ready": 1.0,
}
QUIET = {
    "threat_level": 0.0,
    "distance_ratio": 1.0,
    "health_ratio": 1.0,
    "support_ratio": 1.0,
    "visibility": 0.0,
    "alertness": 0.0,
    "cover_ratio": 1.0,
    "cooldown_ready": 1.0,
}


def test_model_matches_the_declared_schema() -> None:
    model = load_mlp(MODEL_PATH)
    assert model["layers"][0] == len(feature_names("npc_behavior_router"))
    assert model["layers"][-1] == len(LABELS)
    assert model["labels"] == list(LABELS)
    assert model["authority_mode"] == "consultative"


def test_protected_agent_is_vetoed_before_any_model() -> None:
    result = route_npc_behavior({"state": ENGAGING, "constraints": {"noncombatant": True}})
    assert result.output["status"] == "veto"
    assert result.output["behavior"] == "hold_fire"
    assert result.output["micro_nn"] is None
    assert result.output["executes"] is False


def test_zero_health_is_vetoed() -> None:
    state = dict(ENGAGING, health_ratio=0.0)
    result = route_npc_behavior({"state": state, "constraints": {}})
    assert result.output["status"] == "veto"
    assert result.output["behavior"] == "down"


def test_default_engagement_rules_block_the_attack_proposal() -> None:
    result = route_npc_behavior({"state": ENGAGING, "constraints": {}})
    assert result.output["behavior"] != "engage"
    assert result.output["blocked_behaviors"] == ["engage"]
    assert result.output["executes"] is False


def test_free_engagement_rules_allow_a_proposal_only() -> None:
    result = route_npc_behavior(
        {"state": ENGAGING, "constraints": {"rules_of_engagement": "weapons_free"}}
    )
    assert result.output["status"] in {"propose", "abstain"}
    assert result.output["executes"] is False
    if result.output["status"] == "propose":
        assert result.output["behavior"] in LABELS
        assert result.output["certified"] is False


def test_quiet_state_never_proposes_engagement() -> None:
    result = route_npc_behavior(
        {"state": QUIET, "constraints": {"rules_of_engagement": "weapons_free"}}
    )
    assert result.output.get("behavior") != "engage"


def test_unknown_constraint_is_refused() -> None:
    blocked = False
    try:
        route_npc_behavior({"state": ENGAGING, "constraints": {"invisible": True}})
    except ValueError:
        blocked = True
    assert blocked


def test_missing_state_feature_is_not_coerced() -> None:
    partial = dict(ENGAGING)
    partial.pop("visibility")
    blocked = False
    try:
        classify_npc_behavior(partial)
    except ValueError:
        blocked = True
    assert blocked
