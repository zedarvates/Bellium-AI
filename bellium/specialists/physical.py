"""Reference-first physical estimates for game and simulation previews.

The published reference is computed whenever the caller says it is available, and
the cheaper tiers are reported beside it with their live deviation. When the
reference cannot run, the estimate is explicitly provisional and carries the
method, the declared band and the provenance of whatever answered.
"""

from __future__ import annotations

from typing import Any

from bellium.contracts.schema import AuthorityMode, SpecialistResult
from bellium.knn.physical_cases import retrieve_physical_case
from bellium.micro_nn.atmosphere_model import estimate_pressure
from bellium.nano_nn.gravity_residual import estimate_gravity
from bellium.physics.references import (
    hydrostatic_pressure,
    normal_gravity,
    standard_atmosphere,
)

SPECIALIST_ID = "bellium/hybrid/physical-estimate:v0"
FAMILIES = ("gravity", "atmosphere-pressure", "atmosphere-density", "hydrostatic")
REGRESSOR_FAMILIES = ("gravity", "atmosphere-pressure")


def _number(inputs: dict[str, Any], name: str) -> float:
    value = inputs.get(name)
    if value is None:
        raise ValueError(f"input {name} is unknown; do not coerce to 0")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"input {name} must be numeric")
    return float(value)


def _flag(query: dict[str, Any], name: str, default: bool = True) -> bool:
    value = query.get(name, default)
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _reference(family: str, inputs: dict[str, Any]) -> dict[str, Any] | None:
    if family == "gravity":
        estimate = normal_gravity(_number(inputs, "latitude_deg"),
                                  altitude_m=_number(inputs, "altitude_m"))
    elif family in ("atmosphere-pressure", "atmosphere-density"):
        field = "pressure" if family.endswith("pressure") else "density"
        estimate = standard_atmosphere(_number(inputs, "altitude_m"))[field]
    else:
        estimate = hydrostatic_pressure(
            _number(inputs, "depth_m"), density_kg_m3=_number(inputs, "density_kg_m3")
        )
    return estimate.as_dict()


def _approximation(family: str, inputs: dict[str, Any]) -> SpecialistResult | None:
    if family == "gravity":
        return estimate_gravity(
            _number(inputs, "latitude_deg"), altitude_m=_number(inputs, "altitude_m")
        )
    if family == "atmosphere-pressure":
        return estimate_pressure(_number(inputs, "altitude_m"))
    return None


def estimate_physical(query: dict[str, Any]) -> SpecialistResult:
    family = str(query.get("family") or "").strip()
    if family not in FAMILIES:
        return SpecialistResult(
            SPECIALIST_ID,
            {"status": "abstain", "reason": "unknown_family", "family": family,
             "value": None, "alternatives": []},
            0.0, True, AuthorityMode.CONSULTATIVE,
            notes=("Unknown physical family; no value is invented.",),
        )
    inputs = query.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("query needs an inputs object")
    reference_available = _flag(query, "reference_available")
    allow_regressor = _flag(query, "allow_regressor")
    allow_case_table = _flag(query, "allow_case_table")
    alternatives: list[dict[str, Any]] = []
    reference = None
    if reference_available:
        reference = _reference(family, inputs)
    if allow_regressor and family in REGRESSOR_FAMILIES:
        approximation = _approximation(family, inputs)
        if approximation is not None:
            entry = {
                "method": approximation.specialist_id,
                "status": approximation.output["status"],
                "value": approximation.output.get("value"),
                "unit": approximation.output["unit"],
                "deviation_from_reference": approximation.output.get("deviation_from_reference"),
                "declared_band": approximation.output.get(
                    "declared_max_abs", approximation.output.get("declared_max_abs_pa")
                ),
                "heldout_worst_case": approximation.output.get(
                    "heldout_max_abs", approximation.output.get("heldout_max_abs_pa")
                ),
            }
            alternatives.append(entry)
    if allow_case_table:
        table = retrieve_physical_case({"family": family, "inputs": inputs})
        alternatives.append({
            "method": table.specialist_id,
            "status": table.output["status"],
            "value": table.output.get("value"),
            "unit": table.output.get("unit"),
            "support": table.output.get("support"),
            "cases": (table.output.get("bracket") is not None) or table.output.get("support") is not None,
        })
    if reference is not None:
        deviations = [
            alternative["deviation_from_reference"]
            for alternative in alternatives
            if alternative.get("deviation_from_reference") is not None
        ]
        return SpecialistResult(
            SPECIALIST_ID,
            {
                "status": "reference",
                "family": family,
                "value": reference["value"],
                "unit": reference["unit"],
                "method": reference["method"],
                "assumptions": reference["assumptions"],
                "valid_range": reference["valid_range"],
                "provenance": reference["provenance"],
                "uncertainty": reference["uncertainty"],
                "provisional": False,
                "alternatives": alternatives,
                "worst_alternative_deviation": max(deviations) if deviations else None,
            },
            0.95, False, AuthorityMode.CONSULTATIVE,
            notes=(
                "The published reference answered; the cheaper tiers are reported beside it.",
                "No tier is promoted by this result.",
            ),
        )
    candidates = [
        alternative for alternative in alternatives
        if alternative["status"] in ("suggest", "baseline_only")
        and alternative.get("value") is not None
    ]
    if not candidates:
        return SpecialistResult(
            SPECIALIST_ID,
            {"status": "abstain", "reason": "no_available_method", "family": family,
             "value": None, "alternatives": alternatives},
            0.0, True, AuthorityMode.CONSULTATIVE,
            notes=(
                "The reference was declared unavailable and no cheaper tier answered.",
                "Missing inputs, insufficient neighbours and out-of-domain values all abstain.",
            ),
        )
    chosen = candidates[0]
    return SpecialistResult(
        SPECIALIST_ID,
        {
            "status": "approximation",
            "family": family,
            "value": chosen["value"],
            "unit": chosen["unit"],
            "method": chosen["method"],
            "provisional": True,
            "declared_band": chosen.get("declared_band"),
            "published_worst_case": chosen.get("heldout_worst_case"),
            "alternatives": alternatives,
        },
        0.6, False, AuthorityMode.CONSULTATIVE,
        notes=(
            "Provisional estimate from a cheaper tier: the reference was not available.",
            "The published band and worst case are reported so the caller can refuse it.",
        ),
    )
