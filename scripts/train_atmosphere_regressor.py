#!/usr/bin/env python3
"""Train the micro-NN atmosphere pressure approximation with declared thresholds.

The target is the ISA pressure ratio p/p0. Thresholds are fixed before training
and the worst case is converted to pascals for the model card.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bellium.micro_nn.atmosphere_model import (  # noqa: E402
    MODEL_PATH as DEST,
    REFERENCE_ID,
    pressure_input,
)
from bellium.micro_nn.mlp import predict_mlp, save_mlp  # noqa: E402
from bellium.micro_nn.train import regression_metrics, train_regressor  # noqa: E402
from bellium.physics.references import ISA_SEA_LEVEL_PRESSURE_PA, standard_atmosphere  # noqa: E402

LAYERS = [3, 4, 1]
EPOCHS = 1500
LEARNING_RATE = 0.05
MOMENTUM = 0.9
SEED = 43
# Task-driven preview thresholds, declared before training: a pressure error of
# 1 % of sea level is far below the several-percent day-to-day variation that the
# standard atmosphere itself ignores.
RMSE_LIMIT = 0.01
MAX_ABS_LIMIT = 0.03
TARGET_SCALE = 3.0


def samples(offset: int, step: int):
    rows = []
    altitude = offset
    while altitude <= 20000:
        # The logarithm of the pressure ratio is nearly linear in altitude, so a
        # tiny network can fit it far better than the raw exponential ratio.
        ratio = standard_atmosphere(altitude)["pressure"].value / ISA_SEA_LEVEL_PRESSURE_PA
        # Raw target is the log pressure ratio; train_regressor applies the scale.
        rows.append((pressure_input(altitude), math.log(ratio)))
        altitude += step
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="overwrite existing weights")
    options = parser.parse_args()
    if DEST.exists() and not options.force:
        print(f"kept {DEST}")
        return 0
    train = samples(0, 100)
    held = samples(50, 100)
    model = train_regressor(
        train, layer_sizes=LAYERS, epochs=EPOCHS, lr=LEARNING_RATE, seed=SEED,
        target_scale=TARGET_SCALE, momentum=MOMENTUM,
    )
    metrics = regression_metrics(model, held, predict_mlp)
    # A log-ratio error is the relative pressure error for small values.
    rmse_ratio = metrics["rmse"]
    worst_ratio = math.exp(metrics["max_abs"]) - 1.0
    print(f"candidate layers={LAYERS} train={len(train)} held={len(held)}")
    print(
        f"held-out rmse={rmse_ratio:.6f} max_abs={worst_ratio:.6f} of sea level pressure "
        f"(limits rmse<={RMSE_LIMIT:g} max_abs<={MAX_ABS_LIMIT:g})"
    )
    if not math.isfinite(rmse_ratio) or rmse_ratio > RMSE_LIMIT:
        raise SystemExit("atmosphere approximation is outside its declared error band")
    if worst_ratio > MAX_ABS_LIMIT:
        raise SystemExit("atmosphere approximation worst case is outside its declared band")
    model["name"] = "micro_atmosphere_model"
    model["inputs"] = ["altitude_norm", "altitude_norm_squared", "altitude_norm_cubed"]
    model["authority_mode"] = "consultative"
    model["reference_id"] = REFERENCE_ID
    model["log_target"] = True
    model["target_scale"] = TARGET_SCALE
    model["heldout_rmse_ratio"] = round(rmse_ratio, 8)
    model["max_abs_ratio"] = round(worst_ratio, 8)
    model["max_abs_pa"] = round(worst_ratio * ISA_SEA_LEVEL_PRESSURE_PA, 3)
    # Band used for the live abstention check: the task tolerance in pascals.
    model["declared_max_abs_pa"] = round(RMSE_LIMIT * ISA_SEA_LEVEL_PRESSURE_PA, 3)
    save_mlp(DEST, model)
    print(
        f"wrote {DEST} worst case {model['max_abs_pa']:.1f} Pa "
        f"({model['max_abs_pa'] / ISA_SEA_LEVEL_PRESSURE_PA * 100:.3f} % of sea level)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
