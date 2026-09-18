"""Micro-NN atmosphere pressure approximation for constrained runtimes.

The target is the ISA pressure ratio p/p0. The reference model stays
authoritative whenever it can be evaluated.
"""

from __future__ import annotations

import math
from typing import Any

from bellium.contracts.schema import AuthorityMode, SpecialistResult
from bellium.micro_nn.mlp import load_mlp, predict_mlp
from bellium.physics.references import ISA_SEA_LEVEL_PRESSURE_PA, standard_atmosphere
from bellium.resources import model_path

SPECIALIST_ID = "bellium/micro-nn/atmosphere-model:v0"
MODEL_PATH = model_path("micro_nn", "atmosphere-model", "v0.json")
REFERENCE_ID = "bellium/reference/isa"
ALTITUDE_LIMIT_M = 20000.0
TARGET_SCALE = 1.0


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def pressure_input(altitude_m: object) -> list[float]:
    """Ordered model input: normalised altitude and its square and cube."""
    altitude = _number(altitude_m, "altitude_m")
    if not 0.0 <= altitude <= ALTITUDE_LIMIT_M:
        raise ValueError(f"altitude_m must be between 0 and {ALTITUDE_LIMIT_M:g}")
    norm = altitude / ALTITUDE_LIMIT_M
    return [norm, norm * norm, norm * norm * norm]


def estimate_pressure(
    altitude_m: object,
    *,
    model: dict[str, Any] | None = None,
    check_reference: bool = True,
) -> SpecialistResult:
    """Approximate ISA pressure. The live check needs the reference model."""
    vector = pressure_input(altitude_m)
    reference = standard_atmosphere(float(altitude_m))["pressure"]
    output: dict[str, Any] = {
        "unit": "Pa",
        "reference": reference.value,
        "reference_id": REFERENCE_ID,
        "reference_method": reference.method,
        "reference_provenance": reference.provenance,
        "inputs": {"altitude_m": float(altitude_m)},
    }
    if model is None and not MODEL_PATH.exists():
        output.update(status="baseline_only", value=reference.value, model_shipped=False,
                      reason="no_model_shipped")
        return SpecialistResult(
            SPECIALIST_ID, output, 1.0, False, AuthorityMode.CONSULTATIVE,
            notes=("No micro-NN is bundled for this family.", "The ISA model answered instead."),
        )
    loaded = model if model is not None else load_mlp(MODEL_PATH)
    scale = float(loaded.get("target_scale", TARGET_SCALE))
    raw = float(predict_mlp(loaded, vector)[0]) * scale
    ratio = math.exp(raw) if loaded.get("log_target") else raw
    value = ratio * ISA_SEA_LEVEL_PRESSURE_PA
    deviation = abs(value - reference.value)
    tolerance = loaded.get("declared_max_abs_pa")
    tolerance = None if tolerance is None else float(tolerance)
    output.update(
        status="suggest",
        value=value,
        approximation=True,
        deviation_from_reference=round(deviation, 6),
        declared_max_abs_pa=tolerance,
        deviation_checked=bool(check_reference),
        heldout_max_abs_pa=loaded.get("max_abs_pa"),
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
            "Approximation only, for runtimes without the ISA model.",
            "The reference value is reported beside it and stays authoritative.",
        ),
    )
