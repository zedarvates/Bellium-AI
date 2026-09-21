"""Case-based physical retrieval: interpolate only inside a documented domain.

Cases come from the published reference models, not from field measurements.
The specialist interpolates the nearest cases, reports the neighbour support and
the spread, and abstains outside the declared domain or when the neighbours
disagree beyond the tolerance.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from bellium.contracts.schema import AuthorityMode, SpecialistResult
from bellium.resources import model_path

SPECIALIST_ID = "bellium/knn/physical-case-retrieval:v0"
SCHEMA = "bellium.physical-case-memory/v1"
FAMILIES: dict[str, dict[str, Any]] = {
    "gravity": {
        "inputs": ("latitude_deg", "altitude_m"),
        "scales": (90.0, 20000.0),
        "output_unit": "m/s2",
        "file": "gravity-v0.json",
    },
    "atmosphere-pressure": {
        "inputs": ("altitude_m",),
        "scales": (20000.0,),
        "output_unit": "Pa",
        "file": "atmosphere-pressure-v0.json",
    },
    "atmosphere-density": {
        "inputs": ("altitude_m",),
        "scales": (20000.0,),
        "output_unit": "kg/m3",
        "file": "atmosphere-density-v0.json",
    },
    "hydrostatic": {
        "inputs": ("depth_m", "density_kg_m3"),
        "scales": (100.0, 1100.0),
        "output_unit": "Pa",
        "file": "hydrostatic-v0.json",
    },
}
MIN_NEIGHBORS = 3
SUPPORT_RADIUS = 0.25
CONSISTENCY_TOLERANCE = 0.002


def _cases_path(family: str) -> Path:
    return model_path("knn", "physics", FAMILIES[family]["file"])


def _number(name: str, value: object) -> float:
    if value is None:
        raise ValueError(f"input {name} is unknown; do not coerce to 0")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"input {name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"input {name} must be finite")
    return number


def _inputs(family: str, source: object) -> dict[str, float]:
    if not isinstance(source, dict):
        raise ValueError("inputs must be an object")
    spec = FAMILIES[family]
    allowed = set(spec["inputs"])
    missing = [name for name in spec["inputs"] if name not in source]
    extra = [name for name in source if name not in allowed]
    if missing or extra:
        raise ValueError(f"{family}: input mismatch missing={missing} extra={extra}")
    return {name: _number(name, source[name]) for name in spec["inputs"]}


def _vector(family: str, inputs: dict[str, float]) -> list[float]:
    spec = FAMILIES[family]
    return [
        inputs[name] / scale for name, scale in zip(spec["inputs"], spec["scales"])
    ]


def load_cases(family: str, path: str | Path | None = None) -> dict[str, Any]:
    if family not in FAMILIES:
        raise ValueError(f"family must be one of: {', '.join(sorted(FAMILIES))}")
    payload = json.loads(Path(path or _cases_path(family)).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise ValueError("physical case memory schema mismatch")
    if payload.get("family") != family:
        raise ValueError("physical case memory family mismatch")
    domain = payload.get("domain")
    if not isinstance(domain, dict):
        raise ValueError("physical case memory needs a domain")
    for name in FAMILIES[family]["inputs"]:
        bounds = domain.get(name)
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise ValueError(f"domain needs a [min, max] pair for {name}")
    items = payload.get("items")
    if not isinstance(items, list) or len(items) < MIN_NEIGHBORS:
        raise ValueError(f"physical case memory needs at least {MIN_NEIGHBORS} cases")
    loaded = []
    for raw in items:
        if not isinstance(raw, dict):
            raise ValueError("physical case must be an object")
        item_id = str(raw.get("id") or "").strip()
        source = str(raw.get("source") or "").strip()
        if not item_id or not source:
            raise ValueError("physical case needs id and source")
        output = _number("output", raw.get("output"))
        loaded.append({
            "id": item_id,
            "source": source,
            "inputs": _inputs(family, raw.get("inputs")),
            "output": output,
            "vector": _vector(family, _inputs(family, raw.get("inputs"))),
        })
    return {
        "family": family,
        "domain": {name: [float(bounds[0]), float(bounds[1])] for name, bounds in domain.items()},
        "output_unit": str(payload.get("output_unit") or FAMILIES[family]["output_unit"]),
        "reference_method": str(payload.get("reference_method") or ""),
        "notes": str(payload.get("notes") or ""),
        "items": loaded,
    }


def _in_domain(family: str, inputs: dict[str, float], domain: dict[str, list[float]]) -> tuple[bool, dict[str, Any]]:
    evidence = {}
    inside = True
    for name, bounds in domain.items():
        low, high = bounds
        value = inputs[name]
        ok = low <= value <= high
        evidence[name] = {"value": value, "domain": [low, high], "inside": ok}
        inside = inside and ok
    return inside, evidence


def _grid(cases: dict[str, Any], axes: tuple[str, ...]) -> tuple[dict[str, list[float]], dict] | None:
    """Rectangular grid lookup, or None when the table is not a full grid."""
    values = {name: sorted({item["inputs"][name] for item in cases["items"]}) for name in axes}
    lookup = {
        (item["inputs"][axes[0]], item["inputs"][axes[1]]): item for item in cases["items"]
    }
    if len(lookup) != len(cases["items"]):
        return None
    for first in values[axes[0]]:
        for second in values[axes[1]]:
            if (first, second) not in lookup:
                return None
    return values, lookup


def _bracket(values: list[float], target: float) -> tuple[int, int] | None:
    if target < values[0] or target > values[-1]:
        return None
    for index in range(len(values) - 1):
        if values[index] <= target <= values[index + 1]:
            return index, index + 1
    return None


def retrieve_physical_case(
    query: dict[str, Any],
    *,
    memory: dict[str, Any] | None = None,
    k: int = 4,
) -> SpecialistResult:
    family = str(query.get("family") or "").strip()
    if family not in FAMILIES:
        return SpecialistResult(
            SPECIALIST_ID,
            {"status": "abstain", "reason": "unknown_family", "family": family,
             "value": None, "neighbors": []},
            0.0, True, AuthorityMode.CONSULTATIVE,
            notes=("Unknown physical family; no value is invented.",),
        )
    inputs = _inputs(family, query.get("inputs"))
    cases = memory if memory is not None else load_cases(family)
    inside, domain_evidence = _in_domain(family, inputs, cases["domain"])
    if not inside:
        return SpecialistResult(
            SPECIALIST_ID,
            {"status": "abstain", "reason": "out_of_domain", "family": family,
             "value": None, "unit": cases["output_unit"], "domain_evidence": domain_evidence,
             "neighbors": []},
            0.0, True, AuthorityMode.CONSULTATIVE,
            notes=("Interpolation is not offered outside the documented domain.",),
        )
    axes = FAMILIES[family]["inputs"]
    output: dict[str, Any] = {
        "family": family,
        "unit": cases["output_unit"],
        "reference_method": cases["reference_method"],
        "inputs": inputs,
        "neighbors": [],
    }
    if len(axes) == 1:
        axis = axes[0]
        ordered = sorted(cases["items"], key=lambda item: item["inputs"][axis])
        output["method"] = "knn-case-bracketing"
        below = [item for item in ordered if item["inputs"][axis] <= inputs[axis]]
        above = [item for item in ordered if item["inputs"][axis] >= inputs[axis]]
        output["neighbors"] = [
            {"id": item["id"], "inputs": item["inputs"], "output": item["output"],
             "source": item["source"]}
            for item in (below[-1:] + above[:1])
        ]
        if not below or not above:
            output.update(status="abstain", reason="no_bracketing_cases", value=None)
            return SpecialistResult(
                SPECIALIST_ID, output, 0.0, True, AuthorityMode.CONSULTATIVE,
                notes=("The query sits outside the case span of this table.",),
            )
        low, high = below[-1], above[0]
        span = high["inputs"][axis] - low["inputs"][axis]
        if span <= 0.0:
            value = low["output"]
            resolution = 0.0
        else:
            fraction = (inputs[axis] - low["inputs"][axis]) / span
            value = low["output"] + fraction * (high["output"] - low["output"])
            resolution = abs(high["output"] - low["output"]) / max(abs(value), 1e-12)
        output.update(
            status="suggest",
            value=round(value, 6),
            interpolated=True,
            bracket={
                "axis": axis,
                "low": {"input": low["inputs"][axis], "output": low["output"]},
                "high": {"input": high["inputs"][axis], "output": high["output"]},
            },
            local_relative_step=round(resolution, 6),
        )
        confidence = max(0.0, min(0.9, 1.0 - resolution))
        return SpecialistResult(
            SPECIALIST_ID, output, round(confidence, 4), False, AuthorityMode.CONSULTATIVE,
            notes=(
                "Bracketed interpolation between reference-model cases, not a measurement.",
                "The reference formula stays authoritative when it is available.",
            ),
        )
    vector = _vector(family, inputs)
    grid = _grid(cases, axes)
    if grid is not None:
        values, lookup = grid
        brackets = [_bracket(values[name], inputs[name]) for name in axes]
        output["method"] = "knn-case-bilinear"
        if any(bracket is None for bracket in brackets):
            output.update(status="abstain", reason="no_bracketing_cases", value=None)
            return SpecialistResult(
                SPECIALIST_ID, output, 0.0, True, AuthorityMode.CONSULTATIVE,
                notes=("The query sits outside the case span of this table.",),
            )
        (i0, i1), (j0, j1) = brackets  # type: ignore[misc]
        a0, a1 = values[axes[0]][i0], values[axes[0]][i1]
        b0, b1 = values[axes[1]][j0], values[axes[1]][j1]
        corners = {
            (a, b): lookup[(a, b)]["output"] for a in (a0, a1) for b in (b0, b1)
        }
        span_a = a1 - a0
        span_b = b1 - b0
        fa = 0.0 if span_a == 0.0 else (inputs[axes[0]] - a0) / span_a
        fb = 0.0 if span_b == 0.0 else (inputs[axes[1]] - b0) / span_b
        value = (
            corners[(a0, b0)] * (1 - fa) * (1 - fb)
            + corners[(a1, b0)] * fa * (1 - fb)
            + corners[(a0, b1)] * (1 - fa) * fb
            + corners[(a1, b1)] * fa * fb
        )
        outputs = list(corners.values())
        relative_span = (
            0.0 if value == 0.0 else (max(outputs) - min(outputs)) / abs(value)
        )
        output["neighbors"] = [
            {"id": lookup[key]["id"], "inputs": lookup[key]["inputs"],
             "output": lookup[key]["output"], "source": lookup[key]["source"]}
            for key in sorted(corners)
        ]
        output.update(
            status="suggest",
            value=round(value, 6),
            interpolated=True,
            bracket={
                axes[0]: [a0, a1],
                axes[1]: [b0, b1],
            },
            local_relative_span=round(relative_span, 6),
        )
        confidence = max(0.0, min(0.9, 1.0 - relative_span))
        return SpecialistResult(
            SPECIALIST_ID, output, round(confidence, 4), False, AuthorityMode.CONSULTATIVE,
            notes=(
                "Bilinear interpolation on reference-model cases, not a measurement.",
                "The reference formula stays authoritative when it is available.",
            ),
        )
    scored = sorted(
        (
            (
                math.sqrt(sum((vector[i] - item["vector"][i]) ** 2 for i in range(len(vector)))),
                item,
            )
            for item in cases["items"]
        ),
        key=lambda pair: pair[0],
    )[:k]
    support = [pair for pair in scored if pair[0] <= SUPPORT_RADIUS]
    output["method"] = "knn-case-inverse-distance"
    output["neighbors"] = [
        {"id": item["id"], "inputs": item["inputs"], "output": item["output"],
         "distance": round(distance, 6), "source": item["source"]}
        for distance, item in scored
    ]
    output["support"] = len(support)
    if len(support) < MIN_NEIGHBORS:
        output.update(status="abstain", reason="insufficient_neighbor_support", value=None)
        return SpecialistResult(
            SPECIALIST_ID, output, 0.0, True, AuthorityMode.CONSULTATIVE,
            notes=("Too few cases near the query inside the domain.",),
        )

    def interpolate(pairs: list[tuple[float, dict]]) -> float:
        weights = [1.0 / max(distance, 1e-6) ** 2 for distance, _ in pairs]
        total = sum(weights)
        return sum(weight * item["output"] for weight, (_, item) in zip(weights, pairs)) / total

    value = interpolate(support)
    # Leave-one-out check inside the retrieved set: if the nearest case cannot be
    # rebuilt from its neighbours, the local model is not consistent enough.
    held_out = support[0][1]
    rest = support[1:]
    if rest:
        rebuilt = interpolate(rest)
        consistency = abs(rebuilt - held_out["output"]) / max(abs(held_out["output"]), 1e-12)
        output["local_consistency_error"] = round(consistency, 6)
        if consistency > CONSISTENCY_TOLERANCE:
            output.update(status="abstain", reason="neighbors_inconsistent", value=None)
            return SpecialistResult(
                SPECIALIST_ID, output, 0.0, True, AuthorityMode.CONSULTATIVE,
                notes=("The neighbouring cases cannot rebuild each other closely enough.",),
            )
    output.update(status="suggest", value=round(value, 6), interpolated=True)
    confidence = max(
        0.0, min(0.9, 1.0 - 100.0 * float(output.get("local_consistency_error", 0.0)))
    )
    return SpecialistResult(
        SPECIALIST_ID, output, round(confidence, 4), False, AuthorityMode.CONSULTATIVE,
        notes=(
            "Interpolated from reference-model cases, not measured in the field.",
            "The reference formula stays authoritative when it is available.",
        ),
    )
