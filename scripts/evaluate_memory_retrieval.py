"""Authored lexical relevance gate; development/test labels are not field evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bellium.knn.memory_rerank import PreparedMemory, load_memory, rerank_memory  # noqa: E402
from bellium.knn import memory_rerank  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "benchmarks/fixtures/memory-retrieval-corpus-v1.json"
CASES = ROOT / "benchmarks/fixtures/memory-retrieval-cases-v1.json"
THRESHOLDS = {"min_positive_top1_rate": 0.80, "max_negative_suggestions": 0}


def evaluate(memory, cases, profile):
    prepared = PreparedMemory(memory) if profile == "exact" else PreparedMemory(memory, profile=profile)
    rows = []
    times = []
    hits = positive_suggestions = negative_suggestions = relevant_count = returned_count = 0
    positive = negative = 0
    recalls = []
    for case in cases:
        start = time.perf_counter_ns()
        result = rerank_memory({"domain": case["domain"], "terms": case["terms"]}, memory=prepared)
        times.append((time.perf_counter_ns() - start) / 1e6)
        expected = set(case["relevant_ids"])
        neighbors = result.output["neighbors"]
        accepted = not result.abstained
        actual = {item["id"] for item in neighbors} if accepted else set()
        top_correct = accepted and result.output.get("top_id") in expected
        if expected:
            positive += 1
            positive_suggestions += accepted
            hits += top_correct
            recalls.append(len(actual & expected) / len(expected))
        else:
            negative += 1
            negative_suggestions += accepted
        relevant_count += len(actual & expected)
        returned_count += len(actual)
        rows.append({"id": case["id"], "expected_relevant_ids": sorted(expected),
                     "status": result.output["status"], "reason": result.output.get("reason"),
                     "top_correct": top_correct, "returned_ids": [item["id"] for item in neighbors],
                     "abstained": result.abstained})
    positive_rate = hits / positive if positive else None
    passed = positive_rate is not None and positive_rate >= THRESHOLDS["min_positive_top1_rate"]
    passed = passed and negative_suggestions <= THRESHOLDS["max_negative_suggestions"]
    return {
        "profile": profile, "cases": len(cases), "positive_cases": positive,
        "negative_cases": negative, "positive_top1_hits": hits,
        "positive_top1_rate": positive_rate,
        "positive_suggestion_coverage": positive_suggestions / positive if positive else None,
        "mean_accepted_recall_at_5": statistics.mean(recalls) if recalls else None,
        "accepted_neighbor_precision": relevant_count / returned_count if returned_count else None,
        "negative_suggestions": negative_suggestions,
        "negative_abstentions": negative - negative_suggestions,
        "latency_p50_ms": statistics.median(times),
        "latency_p95_ms": sorted(times)[math.ceil(0.95 * len(times)) - 1],
        "gate_passed": passed, "results": rows,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["development", "test", "all"], default="development")
    parser.add_argument("--profiles", nargs="+", choices=["exact", "fr-en"], default=["exact"])
    args = parser.parse_args(argv)
    memory = load_memory(CORPUS)
    payload = json.loads(CASES.read_text(encoding="utf-8"))
    if payload.get("schema") != "bellium.memory-retrieval-cases/v1":
        raise ValueError("evaluation cases schema mismatch")
    seen = set()
    records = {item["id"]: item for item in memory}
    for split in ("development", "test"):
        for case in payload[split]:
            if case["id"] in seen:
                raise ValueError("duplicate evaluation case id")
            seen.add(case["id"])
            for item_id in case["relevant_ids"]:
                if item_id not in records or records[item_id]["domain"] != case["domain"]:
                    raise ValueError("expected relevance must reference a record in the case domain")
    splits = ("development", "test") if args.split == "all" else (args.split,)
    report = {
        "schema": "bellium.memory-retrieval-evaluation/v1", "scope": payload["scope"],
        "corpus_sha256": hashlib.sha256(CORPUS.read_bytes()).hexdigest(),
        "cases_sha256": hashlib.sha256(CASES.read_bytes()).hexdigest(),
        "reranker_source_sha256": hashlib.sha256(Path(memory_rerank.__file__).read_bytes()).hexdigest(),
        "evaluation_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "thresholds_declared_before_evaluation": THRESHOLDS,
        "k": 5, "splits": {split: [evaluate(memory, payload[split], profile)
                                   for profile in args.profiles] for split in splits},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    for split, rows in report["splits"].items():
        for row in rows:
            print(f"{split}/{row['profile']}: positive hits={row['positive_top1_hits']}/"
                  f"{row['positive_cases']}, negative suggestions={row['negative_suggestions']}/"
                  f"{row['negative_cases']}, gate={row['gate_passed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
