"""Measure lexical memory queries on authored corpora; no retrieval-quality claim."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bellium.knn import memory_rerank as reranker  # noqa: E402


def make_memory(size):
    topics = ("image cutout alpha", "mémoire réseau local", "texture tile seam",
              "language phoneme attested", "audio voice recording")
    evidence = ("attested", "reconstructed", "inferred", "speculative")
    return [{"id": f"fixture-{index:07d}", "domain": f"domain-{index % 8}",
             "title": f"Authored fixture {index}",
             "terms": reranker.tokenize(f"{topics[index % 5]} marque{index % 257}"),
             "evidence": evidence[(index // 8) % 4]} for index in range(size)]


QUERIES = {
    "supported": {"domain": "domain-0", "terms": "me\u0301moire re\u0301seau marque7"},
    "unrelated": {"domain": "domain-0", "terms": "unrelated unknownword"},
    "unknown_domain": {"domain": "missing-domain", "terms": "mémoire réseau"},
    "broad_query": {"domain": "domain-0", "terms": "image cutout alpha mémoire réseau local "
                    "texture tile seam language phoneme attested audio voice recording"},
}


def measure(call, repeats):
    call()  # warm-up, outside latency samples
    times = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        result = call()
        times.append((time.perf_counter_ns() - start) / 1e6)
    gc.collect()
    tracemalloc.start()
    try:
        traced_result = call()
        current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    if traced_result != result:
        raise AssertionError("query result changed between latency/allocation runs")
    return {"runs": repeats, "p50_ms": statistics.median(times),
            "p95_ms": sorted(times)[math.ceil(0.95 * len(times)) - 1],
            "peak_traced_bytes": peak, "retained_traced_bytes": current,
            "status": result.output["status"], "neighbors": len(result.output["neighbors"])}, result


def benchmark(sizes, repeats):
    prepared_type = getattr(reranker, "PreparedMemory", None)
    report = {
        "schema": "bellium.memory-rerank-scale/v1",
        "scope": "authored corpus; repeated-query CPU and Python allocations only",
        "python": sys.version, "platform": platform.platform(), "k": 5,
        "query_path": "caller-supplied tokenized list, including validation each call",
        "allocation_scope": "incremental traced Python allocations; not RSS, VRAM or total resident corpus",
        "source_sha256": hashlib.sha256(Path(reranker.__file__).read_bytes()).hexdigest(),
        "cases": [],
    }
    for size in sizes:
        memory = make_memory(size)
        case = {"records": size, "domains": 8, "queries": {}}
        prepared = None
        if prepared_type is not None:
            start = time.perf_counter()
            prepared = prepared_type(memory)
            build_seconds = time.perf_counter() - start
            gc.collect()
            tracemalloc.start()
            try:
                traced_index = prepared_type(memory)
                retained, peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
            del traced_index
            case["preparation"] = {"seconds": build_seconds,
                                   "retained_traced_bytes": retained, "peak_traced_bytes": peak}
        for name, query in QUERIES.items():
            raw_metrics, expected = measure(lambda: reranker.rerank_memory(query, memory=memory), repeats)
            row = {"raw_list": raw_metrics}
            if prepared is not None:
                indexed_metrics, actual = measure(
                    lambda: reranker.rerank_memory(query, memory=prepared), repeats)
                if actual != expected:
                    raise AssertionError(f"prepared query differs: {size}/{name}")
                row["prepared"] = indexed_metrics
                row["exact_result_equivalence"] = True
            case["queries"][name] = row
        report["cases"].append(case)
        print(f"{size} records measured", flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", nargs="+", type=int, default=[1000, 10000, 50000])
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args(argv)
    if args.repeats < 1 or any(size < 1 for size in args.sizes):
        parser.error("sizes and repeats must be positive")
    report = benchmark(args.sizes, args.repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
