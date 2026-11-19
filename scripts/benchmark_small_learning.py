"""Synthetic training and Unicode retrieval gate; no field-quality claims."""

from __future__ import annotations

import argparse
import json
import platform
import random
import re
import statistics
import sys
import time
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bellium.knn.memory_rerank import rerank_memory, tokenize  # noqa: E402
from bellium.micro_nn.mlp import predict_mlp  # noqa: E402
from bellium.micro_nn.train import accuracy, train_classifier  # noqa: E402


def _samples(count, seed, noise=0.0):
    rng = random.Random(seed)
    samples = []
    for _ in range(count):
        vector = [rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0)]
        label = int(sum(vector) > 0.0)
        if rng.random() < noise:
            label = 1 - label
        samples.append((vector, label))
    return samples


def benchmark():
    report = {
        "scope": "synthetic-only; validation selects weights, independent test evaluates them",
        "python": sys.version,
        "platform": platform.platform(),
        "training": [],
        "retrieval": [],
    }
    # Independent generated splits. No fixture or private payload is saved.
    training = _samples(120, 101, noise=0.15)
    validation = _samples(80, 102)
    test = _samples(200, 103)
    for seed in (3, 7, 11):
        row = {"seed": seed, "split_seeds": [101, 102, 103],
               "split_sizes": [120, 80, 200], "training_label_noise": 0.15}
        for name, options in (
            ("fixed", {}),
            ("validation", {"validation_samples": validation, "patience": 8,
                            "min_delta": 0.0001}),
        ):
            start = time.perf_counter()
            model = train_classifier(training, layer_sizes=[2, 6, 2], epochs=80,
                                     lr=0.05, seed=seed, **options)
            row[name] = {
                "seconds": time.perf_counter() - start,
                "test_accuracy": accuracy(model, test, predict_mlp),
                "epochs_run": model.get("training", {}).get("epochs_run", 80),
                "selection": model.get("training"),
                "parameter_count": sum(map(len, model["weights"]))
                                   + sum(map(len, model["biases"])),
            }
        report["training"].append(row)
    report["training_summary"] = {
        name: {
            "mean_test_accuracy": statistics.mean(row[name]["test_accuracy"]
                                                  for row in report["training"]),
            "mean_epochs": statistics.mean(row[name]["epochs_run"]
                                           for row in report["training"]),
        }
        for name in ("fixed", "validation")
    }
    for text in ("image cutout", "mémoire réseau", "acción selección",
                 "μνήμη δίκτυο", "机器 学习"):
        query = unicodedata.normalize("NFD", text)
        old_left = set(re.findall(r"[a-z0-9]+", query.casefold()))
        old_right = set(re.findall(r"[a-z0-9]+", text.casefold()))
        union = old_left | old_right
        legacy_similarity = len(old_left & old_right) / len(union) if union else 0.0
        memory = [{"id": f"fixture-{index}", "domain": "fixture", "title": text,
                   "terms": tokenize(text), "evidence": "attested"} for index in (0, 1)]
        result = rerank_memory({"domain": "fixture", "terms": query}, memory=memory)
        report["retrieval"].append({
            "terms": text, "query_normal_form": "NFD", "memory_normal_form": "NFC",
            "legacy_ascii_similarity": legacy_similarity,
            "legacy_would_have_support": legacy_similarity >= 0.34,
            "unicode_status": result.output["status"],
            "unicode_similarity": result.output["neighbors"][0]["similarity"],
        })
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(benchmark(), indent=2, ensure_ascii=False, allow_nan=False),
                           encoding="utf-8")
    print(args.output)
