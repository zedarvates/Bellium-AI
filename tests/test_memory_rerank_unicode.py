import copy
import json
import random

import pytest

from bellium.knn.memory_rerank import load_memory, rerank_memory, tokenize


def _records():
    return [
        {"id": name, "domain": "bellium", "title": name,
         "terms": tokenize("mémoire réseau"), "evidence": "attested"}
        for name in ("b", "a", "c")
    ]


def test_unicode_tokens_preserve_accents_and_normalize_composed_forms():
    assert tokenize("MÉMOIRE réseau") == {"mémoire", "réseau"}
    assert tokenize("me\u0301moire re\u0301seau") == tokenize("mémoire réseau")
    assert tokenize("机器 学习") == {"机器", "学习"}
    assert tokenize("Straße") == {"strasse"}


def test_equal_scores_have_stable_order_independent_of_memory_order():
    memory = _records()
    before = copy.deepcopy(memory)
    query = {"domain": "bellium", "terms": "mémoire réseau"}
    first = rerank_memory(query, memory=memory, k=2)
    second = rerank_memory(query, memory=list(reversed(memory)), k=2)
    assert [item["id"] for item in first.output["neighbors"]] == ["a", "b"]
    assert first.output == second.output
    assert memory == before


@pytest.mark.parametrize("k", [0, -1, True, 1.5, "2"])
def test_invalid_k_is_rejected(k):
    with pytest.raises(ValueError, match="k"):
        rerank_memory({"domain": "bellium", "terms": "mémoire réseau"}, k=k)


@pytest.mark.parametrize("change", [
    {"body": "forbidden"}, {"evidence": "unknown"}, {"terms": {""}},
])
def test_injected_memory_obeys_loaded_memory_contract(change):
    memory = _records()
    memory[0].update(change)
    with pytest.raises(ValueError):
        rerank_memory({"domain": "bellium", "terms": "mémoire réseau"}, memory=memory)


def test_duplicate_ids_cannot_supply_neighbor_support():
    item = _records()[0]
    with pytest.raises(ValueError, match="duplicate"):
        rerank_memory({"domain": "bellium", "terms": "mémoire réseau"}, memory=[item, item])


def test_loaded_unicode_memory_matches_decomposed_query(tmp_path):
    records = _records()
    for item in records:
        item["terms"] = "mémoire réseau"
    path = tmp_path / "memory.json"
    path.write_text(json.dumps({"schema": "bellium.memory-rerank/v1", "items": records}),
                    encoding="utf-8")
    result = rerank_memory({"domain": "bellium", "terms": "me\u0301moire re\u0301seau"},
                           memory=load_memory(path))
    assert not result.abstained
    assert result.output["neighbors"][0]["similarity"] == 1.0


def test_top_k_matches_full_sort_with_evidence_and_domain_filters():
    rng = random.Random(91)
    terms = {"image", "cutout", "background", "alpha"}
    weights = {"attested": 1.0, "reconstructed": 0.6, "inferred": 0.4, "speculative": 0.2}
    memory = []
    for index in range(200):
        memory.append({
            "id": f"m{index:03d}", "title": f"Fixture {index}",
            "domain": rng.choice(["bellium", "other"]),
            "terms": set(rng.sample(sorted(terms | {"crop", "texture"}), rng.randint(1, 6))),
            "evidence": rng.choice(list(weights)),
        })
    reference = []
    for item in memory:
        sim = len(terms & item["terms"]) / len(terms | item["terms"])
        if item["domain"] == "bellium" and sim >= 0.34:
            reference.append((sim * weights[item["evidence"]], sim, item["id"]))
    reference.sort(key=lambda item: (-item[0], -item[1], item[2]))
    for k in (1, 2, 5, 500):
        result = rerank_memory({"domain": "bellium", "terms": "image cutout background alpha"},
                               memory=memory, k=k)
        assert [item["id"] for item in result.output["neighbors"]] == [
            item[2] for item in reference[:k]
        ]
