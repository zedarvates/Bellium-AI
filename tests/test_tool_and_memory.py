from bellium.knn.memory_rerank import rerank_memory
from bellium.knn.tool_router import retrieve_tool
from bellium.specialists.tool_router import classify_tool_need, route_tool


def _task(**overrides):
    query = {
        "features": {
            "has_code": 0.0,
            "has_files": 0.0,
            "has_error": 0.0,
            "wants_mutation": 0.0,
            "wants_external_state": 0.0,
            "criticality": 0.12,
        },
        "signals": {
            "mentions_secret": False,
            "mentions_deploy": False,
            "mentions_destruction": False,
        },
    }
    query["features"].update(overrides.pop("features", {}))
    query["signals"].update(overrides.pop("signals", {}))
    query.update(overrides)
    return query


def test_deterministic_veto_beats_knn() -> None:
    secret = retrieve_tool(_task(signals={"mentions_secret": True}))
    assert secret.output["tool"] == "escalate"
    assert secret.output["status"] == "rule_escalate"
    mutated = route_tool(_task(features={"wants_mutation": 1.0, "criticality": 0.2}))
    assert mutated.output["tool"] == "escalate"
    assert mutated.output["knn"]["status"] == "rule_escalate"


def test_router_refuses_raw_text() -> None:
    blocked = False
    try:
        retrieve_tool(_task(text="delete production"))
    except ValueError:
        blocked = True
    assert blocked


def test_knn_and_hybrid_choose_inspect_files() -> None:
    query = _task(features={"has_code": 1.0, "has_files": 1.0, "criticality": 0.32})
    knn = retrieve_tool(query)
    assert knn.abstained is False
    assert knn.output["tool"] == "inspect_files"
    nn = classify_tool_need(query)
    assert nn.output["label"] == "use_tool"
    hybrid = route_tool(query)
    assert hybrid.abstained is False
    assert hybrid.output["tool"] == "inspect_files"


def test_router_suggests_none_for_a_plain_question() -> None:
    result = route_tool(_task())
    assert result.output["tool"] == "none"


def test_unknown_requested_tool_escalates() -> None:
    result = retrieve_tool(_task(requested_tool="shell_as_root"))
    assert result.output["tool"] == "escalate"


def test_missing_feature_is_not_coerced_to_zero() -> None:
    query = _task()
    del query["features"]["has_files"]
    blocked = False
    try:
        retrieve_tool(query)
    except ValueError as exc:
        blocked = "has_files" in str(exc)
    assert blocked


def test_memory_reranker_ranks_same_domain_and_isolates_the_rest() -> None:
    result = rerank_memory({"domain": "bellium", "terms": "image cutout background"})
    assert result.abstained is False
    ids = [item["id"] for item in result.output["neighbors"]]
    assert result.output["top_id"] == "mem_cutout"
    assert "mem_white" in ids
    assert "mem_other" not in ids
    other = rerank_memory({"domain": "storycore", "terms": "image panel crop"})
    assert other.output["top_id"] == "mem_other"


def test_memory_reranker_abstains_on_unknown_domain() -> None:
    result = rerank_memory({"domain": "unknown", "terms": "image cutout background"})
    assert result.abstained is True


def test_one_speculative_neighbor_is_not_enough() -> None:
    result = rerank_memory({"domain": "bellium", "terms": "router tool guess sketch"})
    assert result.abstained is True
