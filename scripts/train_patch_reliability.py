#!/usr/bin/env python3
"""Measure whether a nano gate beats the residual rule for normal reliability.

Labels come from the known normals of generated multi-light captures: a patch is
trustworthy when its mean normal error stays inside a declared budget. Both gates
are then compared at equal coverage, which is the only fair comparison when one
of them simply trusts fewer patches. Weights are written only when the candidate
halves the error of the published rule at the same coverage.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bellium.material.controlled import procedural_albedo  # noqa: E402
from bellium.material.photometric import (  # noqa: E402
    light_directions,
    multilight_capture,
    normal_error,
    photometric_normals,
    reliability_features,
    synthetic_geometry,
)
from bellium.micro_nn.mlp import predict_mlp, save_mlp  # noqa: E402
from bellium.micro_nn.train import train_classifier  # noqa: E402
from bellium.nano_nn.contract import NanoBudget, inspect_model  # noqa: E402

DEST = Path(__file__).resolve().parents[1] / "models" / "nano_nn" / "patch-reliability" / "v0.json"
FEATURES = (
    "mean_residual",
    "max_residual",
    "residual_std",
    "residual_gradient",
    "mean_intensity",
    "intensity_range",
)
LAYERS = [len(FEATURES), 4, 2]
EPOCHS = 500
LEARNING_RATE = 0.15
SEED = 61
SIZE = 32
PATCH = 8
ERROR_BUDGET_DEG = 0.5
RULE_LIMIT = 0.02
MARGIN = 0.5
RELIABILITY_BUDGET = NanoBudget(
    max_parameters=64,
    max_model_bytes=8192,
    precision="float-json-weights-v1",
)


def cases(seed: int) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for geometry in ("sphere", "cone", "waves", "tilted-plane"):
        for albedo_kind in ("flat", "checker", "stripes"):
            for specular in (0.0, 0.1, 0.25):
                for ambient in (0.02, 0.08):
                    normals, mask = synthetic_geometry(geometry, size=SIZE)
                    albedo = [
                        [sum(pixel) / 3.0 for pixel in row]
                        for row in procedural_albedo(albedo_kind, size=SIZE, seed=rng.randrange(999))
                    ]
                    lights = light_directions(6)
                    captures = multilight_capture(
                        albedo, normals, lights, ambient=ambient, specular=specular
                    )
                    result = photometric_normals(captures, lights)
                    patches = reliability_features(result["residual"], captures, patch=PATCH)
                    for patch in patches:
                        error = normal_error(
                            result["normals"], normals, mask, indices=patch["indices"]
                        )
                        patch["label"] = int(
                            error["pixels"] > 0
                            and (error["mean_deg"] or 0.0) <= ERROR_BUDGET_DEG
                        )
                        patch["valid"] = error["pixels"] > 0
                        patch["error_deg"] = (
                            error["mean_deg"] if error["mean_deg"] is not None else 999.0
                        )
                    rows.append({
                        "name": f"{geometry}/{albedo_kind}/spec={specular}/amb={ambient}",
                        "patches": patches,
                    })
    return rows


def dataset(rows: list[dict]) -> list[tuple[list[float], int]]:
    samples = []
    for row in rows:
        for patch in row["patches"]:
            samples.append(
                ([patch["features"][name] for name in FEATURES], patch["label"])
            )
    return samples


def rule_gate(features: dict[str, float]) -> bool:
    return features["mean_residual"] <= RULE_LIMIT


def compare(rows: list[dict], model: dict) -> dict:
    """Coverage and error of both gates, then the candidate matched on coverage."""
    rule_trusted: list[float] = []
    nano_scores: list[tuple[float, float]] = []
    for row in rows:
        for patch in row["patches"]:
            if not patch.get("valid", True):
                # Patches outside the surface carry no error to score.
                continue
            vector = [patch["features"][name] for name in FEATURES]
            probability = predict_mlp(model, vector)[1]
            nano_scores.append((probability, patch["error_deg"]))
            if rule_gate(patch["features"]):
                rule_trusted.append(patch["error_deg"])
    total = len(nano_scores)
    rule_coverage = len(rule_trusted) / total
    rule_error = (
        sum(rule_trusted) / len(rule_trusted) if rule_trusted else float("nan")
    )
    ordered = sorted(nano_scores, key=lambda pair: pair[0], reverse=True)
    take = max(1, int(round(rule_coverage * total)))
    nano_trusted = [error for _, error in ordered[:take]]
    nano_error = sum(nano_trusted) / len(nano_trusted)
    return {
        "patches": total,
        "rule_coverage": round(rule_coverage, 4),
        "rule_error_deg": round(rule_error, 4),
        "nano_coverage": round(take / total, 4),
        "nano_error_deg": round(nano_error, 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="overwrite existing weights")
    options = parser.parse_args()
    train_rows = cases(seed=3)
    held_rows = cases(seed=11)
    samples = dataset(train_rows)
    positives = sum(label for _, label in samples)
    print(
        f"cases train={len(train_rows)} held={len(held_rows)} patches={len(samples)} "
        f"trustworthy={positives} ({positives / len(samples):.3f})"
    )
    model = train_classifier(
        samples, layer_sizes=LAYERS, epochs=EPOCHS, lr=LEARNING_RATE, seed=SEED
    )
    contract = inspect_model(model, budget=RELIABILITY_BUDGET)
    comparison = compare(held_rows, model)
    print(f"candidate parameters={contract['parameters']} bytes={contract['model_bytes']}")
    print(
        f"held-out rule: coverage={comparison['rule_coverage']} "
        f"error={comparison['rule_error_deg']}deg | "
        f"nano at same coverage: error={comparison['nano_error_deg']}deg"
    )
    improvement = comparison["rule_error_deg"] - comparison["nano_error_deg"]
    if not math.isfinite(improvement) or (
        comparison["nano_error_deg"] > MARGIN * comparison["rule_error_deg"]
    ):
        print(
            "no model written: the nano gate does not halve the error of the published "
            "residual rule at equal coverage"
        )
        if DEST.exists():
            print(f"existing weights kept untouched at {DEST}")
        return 0
    if DEST.exists() and not options.force:
        print(f"kept {DEST}")
        return 0
    model["name"] = "nano_patch_reliability"
    model["labels"] = ["unreliable", "reliable"]
    model["authority_mode"] = "consultative"
    model["inputs"] = list(FEATURES)
    model["error_budget_deg"] = ERROR_BUDGET_DEG
    model["heldout"] = comparison
    model["precision"] = RELIABILITY_BUDGET.precision
    model["budget"] = RELIABILITY_BUDGET.describe()
    save_mlp(DEST, model)
    print(f"wrote {DEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
