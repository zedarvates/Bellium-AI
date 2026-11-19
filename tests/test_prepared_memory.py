from dataclasses import FrozenInstanceError
import json
import random

import pytest

from bellium.knn import PreparedMemory
from bellium.knn.memory_rerank import load_memory, rerank_memory, tokenize


def _items():
    return [{"id": name, "domain": "bellium", "title": name,
             "terms": tokenize("mémoire réseau"), "evidence": "attested"}
            for name in ("b", "a", "c")]


def test_prepared_snapshot_does_not_follow_source_edits():
    items = _items()
    prepared = PreparedMemory(items)
    query = {"domain": "bellium", "terms": "mémoire réseau"}
    expected = rerank_memory(query, memory=items)
    items[0]["terms"].clear()
    items[1]["title"] = "changed"
    items.clear()
    assert rerank_memory(query, memory=prepared) == expected
    assert len(prepared) == 3
    assert rerank_memory(query, memory=PreparedMemory(items)).abstained


def test_prepared_internal_records_and_postings_are_immutable():
    prepared = PreparedMemory(_items())
    with pytest.raises(FrozenInstanceError):
        prepared._size = 99
    with pytest.raises(TypeError):
        prepared._domains["other"] = prepared._domains["bellium"]
    domain = prepared._domains["bellium"]
    with pytest.raises(TypeError):
        domain.records[0]["title"] = "changed"
    with pytest.raises(AttributeError):
        domain.records[0]["terms"].add("changed")
    with pytest.raises(TypeError):
        domain.postings["mémoire"] = (0,)
    with pytest.raises(FrozenInstanceError):
        domain.records = ()


@pytest.mark.parametrize("change", [
    {"body": "forbidden"}, {"content": "forbidden"}, {"prompt": "forbidden"},
    {"evidence": "unknown"}, {"terms": set()},
])
def test_preparation_preserves_validation_contract(change):
    items = _items()
    items[0].update(change)
    with pytest.raises(ValueError):
        PreparedMemory(items)


def test_preparation_rejects_duplicate_support():
    item = _items()[0]
    with pytest.raises(ValueError, match="duplicate"):
        PreparedMemory([item, item])


def test_prepared_query_does_not_reload_or_revalidate(monkeypatch):
    from bellium.knn import memory_rerank
    prepared = PreparedMemory(_items())
    def forbidden(*args):
        pytest.fail("a prepared query must not load or revalidate")
    monkeypatch.setattr(memory_rerank, "_validated_memory", forbidden)
    monkeypatch.setattr(memory_rerank, "load_memory", forbidden)
    assert not rerank_memory({"domain": "bellium", "terms": "mémoire réseau"},
                             memory=prepared).abstained


def test_prepared_queries_match_list_results_across_domains_tokens_and_k():
    rng = random.Random(82)
    vocabulary = ("mémoire", "réseau", "alpha", "cutout", "background", "texture", "image")
    items = [{"id": f"m{index:03d}", "domain": rng.choice(["a", "b", "c"]),
              "title": f"Fixture {index}",
              "terms": set(rng.sample(vocabulary, rng.randint(1, 7))),
              "evidence": rng.choice(["attested", "inferred", "speculative", "reconstructed"])}
             for index in range(300)]
    prepared = PreparedMemory(items)
    for domain in ("a", "b", "c", "missing"):
        for text in ("image alpha", "me\u0301moire re\u0301seau", "absent", "texture cutout alpha"):
            query = {"domain": domain, "terms": text}
            for k in (1, 2, 5, 500):
                assert rerank_memory(query, memory=prepared, k=k) == rerank_memory(query, memory=items, k=k)


def test_file_changes_require_explicit_snapshot_rebuild(tmp_path):
    items = _items()
    for item in items:
        item["terms"] = "mémoire réseau"
    path = tmp_path / "memory.json"
    path.write_text(json.dumps({"schema": "bellium.memory-rerank/v1", "items": items}),
                    encoding="utf-8")
    prepared = PreparedMemory(load_memory(path))
    query = {"domain": "bellium", "terms": "mémoire réseau"}
    expected = rerank_memory(query, memory=prepared)
    items[0]["title"] = "new title"
    path.write_text(json.dumps({"schema": "bellium.memory-rerank/v1", "items": items}),
                    encoding="utf-8")
    assert rerank_memory(query, memory=prepared) == expected
    rebuilt = PreparedMemory(load_memory(path))
    assert rerank_memory(query, memory=rebuilt) != expected


def test_unknown_and_unrelated_queries_abstain_without_candidate_scan():
    prepared = PreparedMemory(_items())
    assert list(prepared._matching("unknown", {"mémoire"})) == []
    assert list(prepared._matching("bellium", {"unrelated"})) == []
    for query in ({"domain": "unknown", "terms": "mémoire"},
                  {"domain": "bellium", "terms": "unrelated"}):
        result = rerank_memory(query, memory=prepared)
        assert result.abstained
        assert result.output["neighbors"] == []


def test_default_snapshot_preserves_bundled_results():
    query = {"domain": "bellium", "terms": "image cutout background"}
    assert rerank_memory(query, memory=PreparedMemory()) == rerank_memory(query)
