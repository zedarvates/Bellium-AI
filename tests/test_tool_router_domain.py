import pytest

from bellium.knn.tool_router import FEATURE_NAMES
from bellium.specialists import tool_router


def _query():
    return {"features": dict(zip(FEATURE_NAMES, (1.0, 1.0, 0.0, 0.0, 0.0, 0.32)))}


@pytest.mark.parametrize("name", FEATURE_NAMES[:-1])
@pytest.mark.parametrize("value", [0.001, 0.499, 0.5, 0.999])
def test_fractional_flags_abstain_before_loading_model(name, value, monkeypatch):
    query = _query()
    query["features"][name] = value
    monkeypatch.setattr(tool_router, "load_mlp", lambda _: pytest.fail("model must not load"))
    result = tool_router.classify_tool_need(query)
    assert result.abstained
    assert result.confidence == 0.0
    assert result.output["label"] is None
    assert result.output["reason"] == "outside_training_domain"
    assert result.output["unsupported_features"] == [name]


def test_hybrid_abstains_on_fractional_non_veto_feature():
    query = _query()
    query["features"]["has_code"] = 0.8
    result = tool_router.route_tool(query)
    assert result.abstained
    assert result.output["tool"] is None
    assert result.output["micro_nn"]["reason"] == "outside_training_domain"


@pytest.mark.parametrize("feature,value", [("wants_mutation", 0.5), ("criticality", 0.85)])
def test_deterministic_veto_precedes_domain_abstention(feature, value, monkeypatch):
    query = _query()
    query["features"].update({"has_code": 0.8, feature: value})
    monkeypatch.setattr(tool_router, "classify_tool_need", lambda _: pytest.fail("veto must win"))
    result = tool_router.route_tool(query)
    assert not result.abstained
    assert result.output["tool"] == "escalate"
    assert result.output["micro_nn"] is None


def test_boolean_flags_and_continuous_criticality_remain_supported():
    query = _query()
    query["features"]["has_code"] = True
    result = tool_router.classify_tool_need(query)
    assert not result.abstained
    assert result.output["label"] == "use_tool"
