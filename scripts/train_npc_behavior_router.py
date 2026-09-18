#!/usr/bin/env python3
"""Train the bounded NPC behaviour micro-NN.

Labels come from a documented decision rule with a declared label-noise rate.
The network imitates that rule; it is not a learned policy, and it never
receives or emits actions.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bellium.micro_nn.mlp import predict_mlp, save_mlp  # noqa: E402
from bellium.micro_nn.train import accuracy, train_classifier  # noqa: E402

DEST = Path(__file__).resolve().parents[1] / "models" / "micro_nn" / "npc-behavior-router" / "v0.json"
LAYERS = [8, 12, 5]
LABELS = ["idle", "patrol", "investigate", "engage", "retreat"]
NOISE_RATE = 0.05


def rule_label(vector: list[float]) -> int:
    """Weighted danger/safety rule used as the teacher for this classifier."""
    threat, distance, health, support, visibility, alertness, cover, ready = vector
    danger = 0.5 * threat + 0.3 * visibility + 0.2 * (1.0 - distance)
    safety = 0.5 * health + 0.3 * support + 0.2 * cover
    balance = danger - safety
    if balance > 0.35 and health < 0.45:
        return 4
    if balance > 0.15 and ready >= 0.5 and health >= 0.25:
        return 3
    if balance > -0.1:
        return 2
    if alertness > 0.45 or balance > -0.35:
        return 1
    return 0


def make_samples(count: int, seed: int, *, noise: bool) -> list[tuple[list[float], int]]:
    rng = random.Random(seed)
    samples = []
    for _ in range(count):
        vector = [round(rng.random(), 6) for _ in range(LAYERS[0])]
        label = rule_label(vector)
        if noise and rng.random() < NOISE_RATE:
            label = rng.randrange(len(LABELS))
        samples.append((vector, label))
    return samples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="overwrite existing weights")
    options = parser.parse_args()
    if DEST.exists() and not options.force:
        print(f"kept {DEST}")
        return 0
    train = make_samples(900, 17, noise=True)
    held = make_samples(300, 29, noise=False)
    model = train_classifier(train, layer_sizes=LAYERS, epochs=300, lr=0.15, seed=17)
    model["name"] = "npc_behavior_router"
    model["labels"] = LABELS
    model["authority_mode"] = "consultative"
    model["label_noise_rate"] = NOISE_RATE
    score = accuracy(model, held, predict_mlp)
    model["heldout_accuracy"] = round(score, 4)
    if not math.isfinite(score) or score < 0.85:
        raise SystemExit(f"npc behaviour held-out accuracy below 0.85: {score:.3f}")
    save_mlp(DEST, model)
    print(f"wrote {DEST} labels={len(LABELS)} heldout_accuracy={score:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
