from scripts.evaluate_memory_retrieval import evaluate


def _records():
    return [{"id": name, "domain": "d", "title": name, "terms": terms, "evidence": "attested"}
            for name, terms in (("a", "image cutout"), ("b", "image cutout"), ("c", "image patch"))]


def test_partial_abstaining_neighbors_do_not_count_as_accepted_hits():
    cases = [
        {"id": "hit", "domain": "d", "terms": "image cutout", "relevant_ids": ["a", "b"]},
        {"id": "partial", "domain": "d", "terms": "image patch", "relevant_ids": ["c"]},
        {"id": "unrelated", "domain": "d", "terms": "unrelated", "relevant_ids": []},
    ]
    result = evaluate(_records(), cases, "exact")
    assert result["positive_top1_hits"] == 1
    assert result["positive_top1_rate"] == 0.5
    assert result["mean_accepted_recall_at_5"] == 0.5
    assert result["accepted_neighbor_precision"] == 1.0
    assert result["results"][1]["returned_ids"] == ["c"]
    assert result["results"][1]["abstained"]
    assert not result["gate_passed"]


def test_negative_suggestion_fails_gate_even_with_perfect_positive_hits():
    cases = [
        {"id": "positive", "domain": "d", "terms": "image cutout", "relevant_ids": ["a", "b"]},
        {"id": "negative", "domain": "d", "terms": "image cutout", "relevant_ids": []},
    ]
    result = evaluate(_records(), cases, "exact")
    assert result["positive_top1_rate"] == 1.0
    assert result["negative_suggestions"] == 1
    assert result["accepted_neighbor_precision"] == 0.5
    assert not result["gate_passed"]
