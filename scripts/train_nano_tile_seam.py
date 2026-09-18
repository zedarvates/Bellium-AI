#!/usr/bin/env python3
"""Train the nano tile-seam model and measure it against the threshold baseline.

Training vectors are real seam measures computed from generated synthetic
textures. The teacher is an authored perceptual rule, not measured ground
truth, so the reported advantage only holds on this synthetic distribution.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bellium.knn._texture import SEAM_CHANNELS, deterministic_seam_verdict, seam_features  # noqa: E402
from bellium.micro_nn.mlp import predict_mlp, save_mlp  # noqa: E402
from bellium.micro_nn.train import accuracy, train_classifier  # noqa: E402
from bellium.nano_nn.contract import inspect_model  # noqa: E402
from bellium.nano_nn.tile_seam import TILE_SEAM_BUDGET  # noqa: E402

DEST = Path(__file__).resolve().parents[1] / "models" / "nano_nn" / "tile-seam" / "v0.json"
LAYERS = [len(SEAM_CHANNELS), 4, 2]
LABELS = ["continuous", "mismatch"]


def gray(value: float) -> tuple[int, int, int]:
    level = max(0, min(255, int(value)))
    return level, level, level


def texture(seed: int, size: int = 48) -> list[list[tuple[int, int, int]]]:
    """Blend of smooth fields, periodic detail and a wrap discontinuity."""
    rng = random.Random(seed)
    kind = rng.randrange(4)
    step = rng.choice([0.0, 0.0, rng.uniform(4.0, 160.0)])
    period = rng.choice([3, 4, 6, 8, 12, 16])
    amplitude = rng.uniform(0.0, 70.0)
    slope = rng.uniform(-60.0, 60.0)
    rows = []
    for r in range(size):
        row = []
        for c in range(size):
            level = 128 + slope * ((c / size) - 0.5)
            level += amplitude * math.sin(2 * math.pi * c / period)
            if kind == 3:
                level += rng.uniform(-70.0, 70.0)
            if c < size // 2:
                level += step / 2
            else:
                level -= step / 2
            row.append(gray(level))
        rows.append(row)
    return rows


def teacher(features: dict[str, float]) -> int:
    """Authored perceptual rule. Kept explicit so the baseline stays comparable."""
    if features["seam"] <= 0.04:
        return 0
    if features["gap"] > 0.3:
        return 1
    if features["hot_rows"] > 0.5 and features["seam_max"] > 0.25:
        return 1
    if features["interior"] > 0.55 and features["gap"] > 0.15:
        return 1
    return 0


def dataset(count: int, seed: int) -> list[tuple[list[float], int]]:
    samples = []
    for index in range(count):
        features = seam_features(texture(seed * 1000 + index))["x"]
        vector = [float(features[name]) for name in SEAM_CHANNELS]
        samples.append((vector, teacher(features)))
    return samples


def baseline_accuracy(samples: list[tuple[list[float], int]]) -> float:
    hits = 0
    for vector, label in samples:
        features = dict(zip(SEAM_CHANNELS, vector))
        prediction = 0 if deterministic_seam_verdict(features) == "continuous" else 1
        hits += int(prediction == label)
    return hits / len(samples)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="overwrite existing weights")
    options = parser.parse_args()
    if DEST.exists() and not options.force:
        print(f"kept {DEST}")
        return 0
    train = dataset(600, seed=1)
    held = dataset(200, seed=2)
    model = train_classifier(train, layer_sizes=LAYERS, epochs=260, lr=0.15, seed=11)
    model["name"] = "nano_tile_seam"
    model["labels"] = LABELS
    model["authority_mode"] = "consultative"
    model["inputs"] = list(SEAM_CHANNELS)
    nano_accuracy = accuracy(model, held, predict_mlp)
    baseline = baseline_accuracy(held)
    model["heldout_accuracy"] = round(nano_accuracy, 4)
    model["heldout_baseline_accuracy"] = round(baseline, 4)
    model["precision"] = TILE_SEAM_BUDGET.precision
    model["budget"] = TILE_SEAM_BUDGET.describe()
    contract = inspect_model(model, budget=TILE_SEAM_BUDGET)
    if not math.isfinite(nano_accuracy) or nano_accuracy < 0.9:
        raise SystemExit(f"nano held-out accuracy below 0.9: {nano_accuracy:.3f}")
    if nano_accuracy < baseline:
        raise SystemExit(f"nano does not match the baseline: {nano_accuracy:.3f} < {baseline:.3f}")
    save_mlp(DEST, model)
    print(
        f"wrote {DEST} parameters={contract['parameters']} bytes={contract['model_bytes']} "
        f"nano={nano_accuracy:.3f} baseline={baseline:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
