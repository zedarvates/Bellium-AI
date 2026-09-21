#!/usr/bin/env python3
"""Train the bounded consequence micro-NN.

Labels come from the published risk rule in bellium.knn.consequence with a
declared label-noise rate. The network approximates that rule; it never receives
an action and it never executes one. The rule stays the reference, so the
measured gap between the two is reported instead of hidden.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bellium.knn.consequence import (  # noqa: E402
    FEATURE_NAMES,
    LABELS,
    deterministic_risk_label,
)
from bellium.micro_nn.mlp import predict_mlp, save_mlp  # noqa: E402
from bellium.micro_nn.train import accuracy, train_classifier  # noqa: E402

DEST = (
    Path(__file__).resolve().parents[1] / "models" / "micro_nn" / "consequence-predictor" / "v0.json"
)
LAYERS = [len(FEATURE_NAMES), 16, len(LABELS)]
NOISE_RATE = 0.05
TRAIN_SAMPLES = 2000
TRAIN_EPOCHS = 600
TRAIN_LR = 0.2


def make_samples(count: int, seed: int, *, noise: bool) -> list[tuple[list[float], int]]:
    rng = random.Random(seed)
    samples = []
    for _ in range(count):
        vector = [round(rng.random(), 6) for _ in range(len(FEATURE_NAMES))]
        features = dict(zip(FEATURE_NAMES, vector))
        label = LABELS.index(deterministic_risk_label(features))
        if noise and rng.random() < NOISE_RATE:
            label = rng.randrange(len(LABELS))
        samples.append((vector, label))
    return samples


def baseline_accuracy(samples: list[tuple[list[float], int]]) -> float:
    hits = 0
    for vector, label in samples:
        features = dict(zip(FEATURE_NAMES, vector))
        hits += int(LABELS.index(deterministic_risk_label(features)) == label)
    return hits / len(samples)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="overwrite existing weights")
    options = parser.parse_args()
    if DEST.exists() and not options.force:
        print(f"kept {DEST}")
        return 0
    train = make_samples(TRAIN_SAMPLES, seed=23, noise=True)
    held = make_samples(300, seed=31, noise=False)
    model = train_classifier(train, layer_sizes=LAYERS, epochs=TRAIN_EPOCHS, lr=TRAIN_LR, seed=23)
    model["name"] = "consequence_predictor"
    model["labels"] = list(LABELS)
    model["authority_mode"] = "consultative"
    model["inputs"] = list(FEATURE_NAMES)
    model["label_noise_rate"] = NOISE_RATE
    model["reference_rule"] = "bellium/deterministic/consequence-risk:v0"
    score = accuracy(model, held, predict_mlp)
    baseline = baseline_accuracy(held)
    model["heldout_accuracy"] = round(score, 4)
    model["heldout_baseline_accuracy"] = round(baseline, 4)
    model["baseline_note"] = (
        "The published rule generates the held-out labels, so it scores 1.0 by "
        "construction; heldout_accuracy is the network agreement with that rule."
    )
    if not math.isfinite(score) or score < 0.9:
        raise SystemExit(f"consequence held-out accuracy below 0.9: {score:.3f}")
    # The published rule is the reference and is expected to win on its own labels:
    # the gap is recorded rather than treated as a reason to ship the network alone.
    save_mlp(DEST, model)
    print(f"wrote {DEST} labels={len(LABELS)} heldout={score:.3f} baseline={baseline:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
