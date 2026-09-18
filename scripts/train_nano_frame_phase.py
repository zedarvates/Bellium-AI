#!/usr/bin/env python3
"""Measure whether a nano model beats the published animation-phase rule.

Features are real deltas computed from generated motion sequences and the
labels come from the deterministic phase rule. A model is written only when the
trained candidate beats that rule on held-out sequences, so this script is
expected to record a negative result: a threshold rule is smaller, faster and
exact, and a network that merely imitates it should not ship.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bellium.knn._sheet import delta_features, deterministic_phase_verdict  # noqa: E402
from bellium.micro_nn.mlp import predict_mlp, save_mlp  # noqa: E402
from bellium.micro_nn.train import accuracy, train_classifier  # noqa: E402
from bellium.nano_nn.contract import inspect_model  # noqa: E402
from bellium.nano_nn.frame_phase import (  # noqa: E402
    FEATURE_NAMES,
    FRAME_PHASE_BUDGET,
    LABELS,
)

DEST = Path(__file__).resolve().parents[1] / "models" / "nano_nn" / "frame-phase" / "v0.json"
LAYERS = [len(FEATURE_NAMES), 4, 4]
CELL = 24
MARGIN = 0.005

Mask = list[list[int]]


def square(half: int, dx: int = 0, dy: int = 0) -> Mask:
    center = CELL // 2
    return [
        [
            1 if abs(r - center - dy) <= half and abs(c - center - dx) <= half else 0
            for c in range(CELL)
        ]
        for r in range(CELL)
    ]


def disc(radius: int) -> Mask:
    center = CELL // 2
    return [
        [
            1 if (r - center) ** 2 + (c - center) ** 2 <= radius * radius else 0
            for c in range(CELL)
        ]
        for r in range(CELL)
    ]


def ring(radius: int, thickness: int = 1, dx: int = 0, dy: int = 0) -> Mask:
    center = CELL // 2
    mask = [[0] * CELL for _ in range(CELL)]
    for r in range(CELL):
        for c in range(CELL):
            distance = math.hypot(r - center - dy, c - center - dx)
            if radius - thickness <= distance <= radius:
                mask[r][c] = 1
    return mask


def blob(radius: int, dx: int = 0, dy: int = 0) -> Mask:
    center = CELL // 2
    return [
        [
            1 if (r - center - dy) ** 2 + (c - center - dx) ** 2 <= radius * radius else 0
            for c in range(CELL)
        ]
        for r in range(CELL)
    ]


def sequences(seed: int) -> list[list[Mask]]:
    """Motion profiles covering holds, ramps, peaks and area-dominant changes."""
    rng = random.Random(seed)
    size = rng.choice([3, 4, 5])
    return [
        [square(size), square(size), square(size)],
        [square(size), square(size, dx=1), square(size, dx=2), square(size, dx=4)],
        [square(size), square(size, dx=6), square(size, dx=6), square(size, dx=6)],
        [square(size, dx=rng.choice([5, 6, 7])), square(size, dx=3), square(size, dx=1), square(size)],
        [ring(4), ring(5), ring(6), ring(7), ring(8)],
        [ring(9), ring(7), ring(5), ring(4)],
        [ring(5), ring(6, dx=2), ring(7, dx=3), ring(8, dx=5)],
        [ring(6, dx=3), ring(5, dx=2), ring(4, dx=1)],
        [blob(6), blob(7, dx=1, dy=1), blob(8, dx=2, dy=2), blob(9, dx=3, dy=3)],
        [disc(rng.choice([4, 5])), disc(rng.choice([7, 8])), disc(rng.choice([4, 5])),
         disc(rng.choice([7, 8]))],
        [blob(5), blob(6, dx=1), blob(7, dx=2), blob(8, dx=3)],
        [blob(8, dx=3), blob(6, dx=1), blob(5), blob(5)],
        [square(size, dx=2, dy=1), square(size, dx=3, dy=2), square(size, dx=1, dy=1)],
    ]


def dataset(count: int, seed: int) -> list[tuple[list[float], int]]:
    samples: list[tuple[list[float], int]] = []
    round_index = 0
    while len(samples) < count:
        for sequence in sequences(seed * 1000 + round_index):
            previous_change = None
            for index in range(1, len(sequence)):
                position = index / (len(sequence) - 1)
                features = delta_features(
                    sequence[index - 1], sequence[index], position, previous_change
                )
                samples.append(
                    (
                        [float(features[name]) for name in FEATURE_NAMES],
                        LABELS.index(deterministic_phase_verdict(features)),
                    )
                )
                previous_change = features["changed_ratio"]
        round_index += 1
    return samples[:count]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="replace an existing model")
    options = parser.parse_args()
    train = dataset(600, seed=1)
    held = dataset(200, seed=2)
    model = train_classifier(train, layer_sizes=LAYERS, epochs=300, lr=0.15, seed=13)
    contract = inspect_model(model, budget=FRAME_PHASE_BUDGET)
    nano_accuracy = accuracy(model, held, predict_mlp)
    baseline_accuracy = 1.0
    print(
        f"candidate layers={LAYERS} parameters={contract['parameters']} "
        f"bytes={contract['model_bytes']}"
    )
    print(f"held-out agreement with the rule: candidate={nano_accuracy:.3f} "
          f"rule={baseline_accuracy:.3f}")
    if not math.isfinite(nano_accuracy) or nano_accuracy <= baseline_accuracy + MARGIN:
        print(
            "no model written: the deterministic phase rule is not beaten, so it stays "
            "the answer and the tier reports baseline_only"
        )
        if DEST.exists():
            print(f"existing weights kept untouched at {DEST}")
        return 0
    if DEST.exists() and not options.force:
        print(f"kept {DEST}")
        return 0
    model["name"] = "nano_frame_phase"
    model["labels"] = list(LABELS)
    model["authority_mode"] = "consultative"
    model["inputs"] = list(FEATURE_NAMES)
    model["heldout_accuracy"] = round(nano_accuracy, 4)
    model["precision"] = FRAME_PHASE_BUDGET.precision
    model["budget"] = FRAME_PHASE_BUDGET.describe()
    save_mlp(DEST, model)
    print(f"wrote {DEST} agreement={nano_accuracy:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
