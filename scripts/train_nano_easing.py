#!/usr/bin/env python3
"""Measure whether a nano model beats the published easing rule on noisy curves.

Both tiers do the same job: name the easing curve of a motion series from seven
progress samples. The rule applies midpoint and overshoot thresholds; the
candidate is a small network trained on noisy samples. Weights are written only
when the candidate beats the rule by the declared margin, so a negative result
is a normal outcome of this script.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bellium.knn._motion import (  # noqa: E402
    REFERENCE_CURVES,
    SAMPLES,
    deterministic_easing_verdict,
)
from bellium.micro_nn.mlp import predict_mlp, save_mlp  # noqa: E402
from bellium.micro_nn.train import accuracy, train_classifier  # noqa: E402
from bellium.nano_nn.contract import NanoBudget, inspect_model  # noqa: E402

DEST = Path(__file__).resolve().parents[1] / "models" / "nano_nn" / "easing-curve" / "v0.json"
LABELS = tuple(sorted(REFERENCE_CURVES))
LAYERS = [SAMPLES, 4, len(LABELS)]
EPOCHS = 600
LEARNING_RATE = 0.15
SEED = 51
NOISE_LEVELS = (0.02, 0.05, 0.08)
TRAIN_NOISE = 0.05
MARGIN = 0.02
EASING_BUDGET = NanoBudget(
    max_parameters=64,
    max_model_bytes=8192,
    precision="float-json-weights-v1",
)


def noisy_curve(rng: random.Random, noise: float) -> tuple[list[float], int]:
    label_index = rng.randrange(len(LABELS))
    curve = REFERENCE_CURVES[LABELS[label_index]]
    vector = [
        min(max(value + rng.gauss(0.0, noise), 0.0), 1.5) for value in curve
    ]
    return vector, label_index


def samples(count: int, seed: int, noise: float) -> list[tuple[list[float], int]]:
    rng = random.Random(seed)
    return [noisy_curve(rng, noise) for _ in range(count)]


def rule_accuracy(rows: list[tuple[list[float], int]]) -> float:
    hits = 0
    for vector, label in rows:
        verdict = deterministic_easing_verdict(list(vector))
        hits += int(verdict == LABELS[label])
    return hits / len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="overwrite existing weights")
    options = parser.parse_args()
    train = samples(1500, seed=7, noise=TRAIN_NOISE)
    model = train_classifier(
        train, layer_sizes=LAYERS, epochs=EPOCHS, lr=LEARNING_RATE, seed=SEED
    )
    contract = inspect_model(model, budget=EASING_BUDGET)
    print(
        f"candidate layers={LAYERS} parameters={contract['parameters']} "
        f"bytes={contract['model_bytes']} trained on noise={TRAIN_NOISE}"
    )
    nano_total = 0
    rule_total = 0
    evaluated = 0
    per_level: dict[str, dict[str, float]] = {}
    for noise in NOISE_LEVELS:
        held = samples(300, seed=100 + int(noise * 1000), noise=noise)
        nano = accuracy(model, held, predict_mlp)
        rule = rule_accuracy(held)
        per_level[f"{noise:.2f}"] = {"nano": round(nano, 4), "rule": round(rule, 4)}
        nano_total += nano * len(held)
        rule_total += rule * len(held)
        evaluated += len(held)
        print(f"noise={noise:.2f} held={len(held)} nano={nano:.3f} rule={rule:.3f}")
    nano_accuracy = nano_total / evaluated
    rule_accuracy_score = rule_total / evaluated
    print(
        f"overall nano={nano_accuracy:.4f} rule={rule_accuracy_score:.4f} "
        f"(margin required {MARGIN:g})"
    )
    if (
        not math.isfinite(nano_accuracy)
        or nano_accuracy < rule_accuracy_score + MARGIN
    ):
        print(
            "no model written: the published easing rule is not beaten under noise, "
            "so it stays the answer and the tier reports baseline_only"
        )
        if DEST.exists():
            print(f"existing weights kept untouched at {DEST}")
        return 0
    if DEST.exists() and not options.force:
        print(f"kept {DEST}")
        return 0
    model["name"] = "nano_easing_curve"
    model["labels"] = list(LABELS)
    model["authority_mode"] = "consultative"
    model["inputs"] = [f"s{index}" for index in range(SAMPLES)]
    model["train_noise_sigma"] = TRAIN_NOISE
    model["heldout_accuracy"] = round(nano_accuracy, 4)
    model["heldout_rule_accuracy"] = round(rule_accuracy_score, 4)
    model["heldout_by_noise"] = per_level
    model["precision"] = EASING_BUDGET.precision
    model["budget"] = EASING_BUDGET.describe()
    save_mlp(DEST, model)
    print(f"wrote {DEST} accuracy={nano_accuracy:.3f} rule={rule_accuracy_score:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
