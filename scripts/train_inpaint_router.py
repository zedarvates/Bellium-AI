#!/usr/bin/env python3
"""Train the native inpaint-router micro-NN on synthetic hole statistics."""

from __future__ import annotations

import random
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bellium.micro_nn.mlp import predict_mlp, save_mlp
from bellium.micro_nn.train import accuracy, train_classifier

DEST = Path(__file__).resolve().parents[1] / "models" / "micro_nn" / "inpaint-router" / "v0.json"


def _label(area: float, std: float, span: float, border: float, holes: float) -> int:
    if area <= 0.08 and std <= 0.18 and span <= 0.30 and holes <= 0.4:
        return 0  # patch_knn
    if area >= 0.18 or std >= 0.40 or span >= 0.50 or holes >= 0.8:
        return 1  # escalate
    if border and area >= 0.10:
        return 1
    return 0 if std < 0.22 and area < 0.12 else 1


def make_samples(n: int, rng: random.Random) -> list[tuple[list[float], int]]:
    samples = []
    for _ in range(n):
        area = rng.random() * 0.35
        fill = 0.3 + rng.random() * 0.7
        border = 1.0 if rng.random() < 0.3 else 0.0
        std = rng.random() * 0.7
        holes = rng.random()
        span = min(1.0, area * (1.4 + rng.random()) + rng.random() * 0.2)
        label = _label(area, std, span, border, holes)
        samples.append(([area, fill, border, std, holes, span], label))
    return samples


def main() -> int:
    rng = random.Random(11)
    train = make_samples(800, rng)
    held = make_samples(200, random.Random(99))
    model = train_classifier(train, layer_sizes=[6, 8, 2], epochs=250, lr=0.12, seed=11)
    model["name"] = "inpaint_router"
    model["labels"] = ["patch_knn", "escalate"]
    model["authority_mode"] = "consultative"
    acc = accuracy(model, held, predict_mlp)
    model["heldout_accuracy"] = round(acc, 4)
    if not math.isfinite(acc) or acc < 0.85:
        raise SystemExit("inpaint router held-out accuracy below 0.85")
    save_mlp(DEST, model)
    print(f"wrote {DEST} heldout_accuracy={acc:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
