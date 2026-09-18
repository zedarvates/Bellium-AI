#!/usr/bin/env python3
from __future__ import annotations

import random
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bellium.micro_nn.mlp import predict_mlp, save_mlp
from bellium.micro_nn.train import accuracy, train_classifier

DEST = Path(__file__).resolve().parents[1] / "models" / "micro_nn" / "tool-router" / "v0.json"


def _label(has_code: float, has_files: float, has_error: float, mutation: float, external: float, crit: float) -> int:
    if mutation >= 0.5 or crit >= 0.85:
        return 2
    if has_code >= 0.5 or has_files >= 0.5 or has_error >= 0.5 or external >= 0.5:
        return 1
    return 0


def make_samples(n: int, rng: random.Random) -> list[tuple[list[float], int]]:
    samples = []
    for _ in range(n):
        vec = [1.0 if rng.random() < 0.4 else 0.0 for _ in range(5)]
        crit = rng.random()
        vec.append(crit)
        samples.append((vec, _label(*vec)))
    return samples


def main() -> int:
    train = make_samples(900, random.Random(5))
    held = make_samples(200, random.Random(21))
    model = train_classifier(train, layer_sizes=[6, 8, 3], epochs=220, lr=0.12, seed=5)
    model["name"] = "tool_router"
    model["labels"] = ["none", "use_tool", "escalate"]
    model["authority_mode"] = "consultative"
    acc = accuracy(model, held, predict_mlp)
    model["heldout_accuracy"] = round(acc, 4)
    if not math.isfinite(acc) or acc < 0.85:
        raise SystemExit("tool router held-out accuracy below 0.85")
    save_mlp(DEST, model)
    print(f"wrote {DEST} heldout_accuracy={acc:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
