#!/usr/bin/env python3
"""Train the nano gravity approximation and enforce a declared error band.

The acceptance thresholds are fixed here, before training. Weights are written
only when the candidate stays inside them on a held-out latitude/altitude grid.
The reference formula remains the authority; this model exists for runtimes that
cannot evaluate it.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bellium.micro_nn.mlp import predict_mlp, save_mlp  # noqa: E402
from bellium.micro_nn.train import regression_metrics, train_regressor  # noqa: E402
from bellium.nano_nn.contract import inspect_model  # noqa: E402
from bellium.nano_nn.gravity_residual import (  # noqa: E402
    GRAVITY_RESIDUAL_BUDGET,
    REFERENCE_ID,
    gravity_input,
)
from bellium.physics.references import normal_gravity  # noqa: E402
from bellium.physics.references import STANDARD_GRAVITY  # noqa: E402

DEST = Path(__file__).resolve().parents[1] / "models" / "nano_nn" / "gravity-residual" / "v0.json"
LAYERS = [3, 1]
EPOCHS = 400
LEARNING_RATE = 0.05
MOMENTUM = 0.0
SEED = 41
# Task-driven preview thresholds, declared before training: a gravity error of
# 1e-2 m/s2 (about 1 mGal) moves a 1.27 m jump apex by roughly 1.3 mm, which is
# far below anything a game or a simulation preview can act on.
# Task-driven preview thresholds, declared before training: an error of
# 1e-3 m/s2 (0.1 mGal) moves a 1.27 m jump apex by roughly 0.13 mm, far below
# anything a game or a simulation preview can act on.
RMSE_LIMIT = 1e-3
MAX_ABS_LIMIT = 5e-3
RESIDUAL_SCALE = 0.05


def samples(latitude_step: float, latitude_offset: float, altitude_offset: int,
            altitude_step: int = 1000):
    rows = []
    latitude = -90.0 + latitude_offset
    while latitude <= 90.0:
        altitude = altitude_offset
        while altitude <= 20000:
            # The target is the deviation from standard gravity, rescaled: the
            # signal is only about 0.5 % of g, so a raw target would be learned
            # as a constant.
            # Raw target in m/s2: train_regressor applies RESIDUAL_SCALE once, and
            # regression_metrics undoes it, so the reported errors are physical.
            value = normal_gravity(latitude, altitude_m=altitude).value - STANDARD_GRAVITY
            rows.append((gravity_input(latitude, altitude), value))
            altitude += altitude_step
        latitude += latitude_step
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="overwrite existing weights")
    options = parser.parse_args()
    if DEST.exists() and not options.force:
        print(f"kept {DEST}")
        return 0
    train = samples(6.0, 0.0, 0, altitude_step=1000)
    held = samples(3.0, 1.5, 500, altitude_step=1000)
    model = train_regressor(
        train, layer_sizes=LAYERS, epochs=EPOCHS, lr=LEARNING_RATE, seed=SEED,
        target_scale=RESIDUAL_SCALE, momentum=MOMENTUM,
    )
    metrics = regression_metrics(model, held, predict_mlp)
    contract = inspect_model(model, budget=GRAVITY_RESIDUAL_BUDGET)
    print(
        f"candidate parameters={contract['parameters']} bytes={contract['model_bytes']} "
        f"train={len(train)} held={len(held)}"
    )
    rmse = metrics["rmse"]
    worst = metrics["max_abs"]
    print(
        f"held-out rmse={rmse:.6f} m/s2 max_abs={worst:.6f} m/s2 "
        f"bias={metrics['bias']:+.6f} "
        f"(limits rmse<={RMSE_LIMIT:g} max_abs<={MAX_ABS_LIMIT:g})"
    )
    if not math.isfinite(rmse) or rmse > RMSE_LIMIT:
        raise SystemExit("gravity approximation is outside its declared error band")
    if worst > MAX_ABS_LIMIT:
        raise SystemExit("gravity approximation worst case is outside its declared band")
    model["name"] = "nano_gravity_residual"
    model["inputs"] = ["latitude_norm", "altitude_norm", "latitude_norm_squared"]
    model["authority_mode"] = "consultative"
    model["reference_id"] = REFERENCE_ID
    model["offset"] = STANDARD_GRAVITY
    model["target_scale"] = RESIDUAL_SCALE
    model["heldout_rmse"] = round(rmse, 8)
    model["max_abs"] = round(worst, 8)
    # Band used for the live abstention check: the task tolerance rather than the
    # measured worst case, so the check answers "good enough to preview?".
    model["declared_max_abs"] = RMSE_LIMIT
    model["max_abs_pa"] = None
    model["precision"] = GRAVITY_RESIDUAL_BUDGET.precision
    model["budget"] = GRAVITY_RESIDUAL_BUDGET.describe()
    save_mlp(DEST, model)
    print(f"wrote {DEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
