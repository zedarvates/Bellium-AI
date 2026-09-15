"""Reproducible fitting and honest, held-out metrics for all four mechanisms."""
from __future__ import annotations

import platform
import hashlib
import statistics
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path

import PIL

from .dataset import dataset_digest, validate_splits
from .features import FEATURE_NAMES, rule_advice
from .models import Predictors, canonical_bytes, train_bundle


def _metrics(cases: list[dict]) -> dict:
    total = len(cases)
    answered = [row for row in cases if row["predicted"] != "abstain"]
    correct = sum(row["predicted"] == row["expected"] for row in answered)
    defects = sum(row["expected"] == "review" for row in cases)
    missed = sum(row["expected"] == "review" and row["predicted"] == "candidate" for row in cases)
    latencies = sorted(row["latency_ms"] for row in cases)
    return {
        "cases": total, "answered": len(answered), "correct": correct,
        "coverage": len(answered) / total if total else None,
        "selective_accuracy": correct / len(answered) if answered else None,
        "correct_over_all_cases": correct / total if total else None,
        "review_cases": defects, "false_candidates": missed,
        "false_candidate_rate": missed / defects if defects else None,
        "abstentions": total - len(answered),
        "review_or_abstain_rate": sum(row["predicted"] != "candidate" for row in cases) / total if total else None,
        "latency_p50_ms": statistics.median(latencies) if latencies else None,
        "latency_p95_ms": latencies[max(0, (95 * total + 99) // 100 - 1)] if total else None,
        "confusion": {truth: {prediction: sum(row["expected"] == truth and row["predicted"] == prediction
                                                for row in cases)
                               for prediction in ("candidate", "review", "abstain")}
                      for truth in ("candidate", "review")},
    }


def run_benchmark(rows: list[dict], *, provenance: str, seed: int = 17, epochs: int = 100) -> tuple[dict, dict]:
    validate_splits(rows)
    train = [row for row in rows if row["split"] == "train"]
    # Collapse exact duplicates before training; repeated files cannot add evidence.
    unique = {row["sha256"]: row for row in train}
    started = time.perf_counter()
    bundle = train_bundle(list(unique.values()), seed=seed, epochs=epochs)
    fit_ms = (time.perf_counter() - started) * 1000
    predictors = Predictors(bundle)
    results = {}
    for name in ("rules", "knn", "nano", "micro"):
        cases = []
        for row in rows:
            if row["split"] == "train":
                continue
            started = time.perf_counter()
            prediction = rule_advice(row["features"]) if name == "rules" else predictors.predict(
                name, row["features"], sha256=row["sha256"])
            cases.append({"id": row["id"], "split": row["split"], "reason": row["reason"],
                          "expected": row["label"], "predicted": prediction["label"],
                          "latency_ms": (time.perf_counter() - started) * 1000})
        # Measure Python allocation separately so tracing does not distort latency.
        probe = next(row for row in rows if row["split"] == "test")
        tracemalloc.start()
        try:
            if name == "rules":
                rule_advice(probe["features"])
            else:
                predictors.predict(name, probe["features"])
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        model_bytes = 0 if name == "rules" else len(canonical_bytes(
            bundle["support"] if name == "knn" else bundle["networks"][name]))
        results[name] = {
            "parameters": {"rules": 0, "knn": 0, "nano": 7, "micro": 65}[name],
            "serialized_model_bytes": model_bytes,
            "shared_support_bytes": len(canonical_bytes(bundle["support"])) if name in ("nano", "micro") else 0,
            "python_prediction_allocation_peak_bytes": peak,
            "by_split": {split: _metrics([case for case in cases if case["split"] == split])
                         for split in ("validation", "test", "ood")},
            "cases": cases,
        }
    report = {
        "schema": "bellium.alpha-benchmark/v1", "authority": "shadow",
        "implementation_hash_encoding": "UTF-8 with newlines normalized to LF",
        "implementation_sha256": {path.name: hashlib.sha256(path.read_text(encoding="utf-8").encode()).hexdigest()
                                  for path in sorted(Path(__file__).parent.glob("*.py"))},
        "created_at": datetime.now(timezone.utc).isoformat(),
        "provenance": provenance, "dataset_sha256": dataset_digest(rows),
        "bundle_sha256": bundle["bundle_sha256"], "training_digest": bundle["training_digest"],
        "feature_names": list(FEATURE_NAMES), "seed": seed, "epochs": epochs,
        "split_counts": {split: sum(row["split"] == split for row in rows)
                         for split in ("train", "validation", "test", "ood")},
        "unique_training_assets": len(unique), "fit_ms": fit_ms,
        "feature_extraction_p50_ms": statistics.median(row["feature_ms"] for row in rows),
        "hardware": {"system": platform.system(), "machine": platform.machine(),
                     "processor": platform.processor() or "not_reported",
                     "python": platform.python_version(), "pillow": PIL.__version__,
                     "device": "cpu", "gpu_used": False, "vram_bytes": 0},
        "threshold_selection": "fixed a priori; validation reported, no test tuning",
        "limitations": [
            "Alpha geometry only; no RGB semantics, aesthetics, mesh, licensing or publication proof.",
            "Synthetic fixtures are implementation evidence, not real Asset Factory quality evidence.",
            "candidate means no detected matte defect; it never authorizes production or upload.",
            "Scores and training-distance abstention are uncalibrated heuristics.",
            "Latency includes support search, excludes decoding/features/loading; one call per case.",
            "Allocation peak is a one-case Python allocation probe, not total process RAM.",
            "Networks also need the shared support ledger for abstention; weight bytes are not total memory.",
        ],
        "models": results,
    }
    return report, bundle
