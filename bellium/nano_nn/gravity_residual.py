"""Nano-NN gravity approximation for runtimes that cannot evaluate the reference.

The three inputs use the physical basis of the relation: sin^2(latitude),
sin^4(latitude) and normalised altitude. With that basis a four-parameter linear
model reproduces the WGS-84 normal gravity plus free-air term far better than the
ReLU networks of the same budget. The reference formula stays authoritative: when
it is available the result reports it beside the approximation and publishes the
live deviation.
"""

from __future__ import annotations

import math
from typing import Any

from bellium.contracts.schema import AuthorityMode, SpecialistResult
from bellium.micro_nn.mlp import load_mlp, predict_mlp
from bellium.nano_nn.contract import NanoBudget, inspect_model
from bellium.physics.references import normal_gravity
from bellium.resources import model_path

SPECIALIST_ID = "bellium/nano-nn/gravity-residual:v0"
MODEL_PATH = model_path("nano_nn", "gravity-residual", "v0.json")
REFERENCE_ID = "bellium/reference/wgs84-somigliana"
TARGET_SCALE = 10.0
ALTITUDE_LIMIT_M = 20000.0
GRAVITY_RESIDUAL_BUDGET = NanoBudget(
    max_parameters=64,
    max_model_bytes=8192,
    precision="float-json-weights-v1",
)


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def gravity_input(latitude_deg: object, altitude_m: object = 0.0) -> list[float]:
    """Ordered model input: sin^2(latitude), sin^4(latitude), normalised altitude."""
    latitude = _number(latitude_deg, "latitude_deg")
    if not -90.0 <= latitude <= 90.0:
        raise ValueError("latitude_deg must be between -90 and 90")
    altitude = _number(altitude_m, "altitude_m")
    if not 0.0 <= altitude <= ALTITUDE_LIMIT_M:
        raise ValueError(f"altitude_m must be between 0 and {ALTITUDE_LIMIT_M:g}")
    squared = math.sin(math.radians(latitude)) ** 2
    return [squared, squared * squared, altitude / ALTITUDE_LIMIT_M]


def estimate_gravity(
    latitude_deg: object,
    *,
    altitude_m: object = 0.0,
    model: dict[str, Any] | None = None,
    check_reference: bool = True,
) -> SpecialistResult:
    """Approximate gravity. The live deviation check needs the reference formula.

    The check_reference argument is the honest default: the approximation
    compares itself with the reference and abstains outside its declared band. A
    runtime that cannot evaluate the reference passes False and inherits the
    published band from the model card instead.
    """
    vector = gravity_input(latitude_deg, altitude_m)
    reference = normal_gravity(float(latitude_deg), altitude_m=float(altitude_m))
    output: dict[str, Any] = {
        "unit": "m/s2",
        "reference": reference.value,
        "reference_id": REFERENCE_ID,
        "reference_method": reference.method,
        "reference_provenance": reference.provenance,
        "inputs": {"latitude_deg": float(latitude_deg), "altitude_m": float(altitude_m)},
    }
    if model is None and not MODEL_PATH.exists():
        output.update(status="baseline_only", value=reference.value, model_shipped=False,
                      reason="no_model_shipped")
        return SpecialistResult(
            SPECIALIST_ID, output, 1.0, False, AuthorityMode.CONSULTATIVE,
            notes=(
                "No nano model is bundled for this family.",
                "The reference formula answered instead.",
            ),
        )
    loaded = model if model is not None else load_mlp(MODEL_PATH)
    contract = inspect_model(loaded, budget=GRAVITY_RESIDUAL_BUDGET)
    scale = float(loaded.get("target_scale", TARGET_SCALE))
    offset = float(loaded.get("offset", 0.0))
    value = offset + float(predict_mlp(loaded, vector)[0]) * scale
    deviation = abs(value - reference.value)
    tolerance = loaded.get("declared_max_abs")
    tolerance = None if tolerance is None else float(tolerance)
    output.update(
        status="suggest",
        value=value,
        approximation=True,
        deviation_from_reference=round(deviation, 8),
        declared_max_abs=tolerance,
        deviation_checked=bool(check_reference),
        heldout_max_abs=loaded.get("max_abs"),
        contract=contract,
        provisional=True,
    )
    if check_reference and tolerance is not None and deviation > tolerance:
        output.update(status="abstain", reason="outside_declared_error_band", value=None)
        return SpecialistResult(
            SPECIALIST_ID, output, 0.0, True, AuthorityMode.CONSULTATIVE,
            evidence_ref=MODEL_PATH.name,
            notes=("The approximation left its declared error band on this input.",),
        )
    return SpecialistResult(
        SPECIALIST_ID, output, 0.8, False, AuthorityMode.CONSULTATIVE,
        evidence_ref=MODEL_PATH.name,
        notes=(
            "Approximation only, for runtimes without the reference formula.",
            "The reference value is reported beside it and stays authoritative.",
        ),
    )
