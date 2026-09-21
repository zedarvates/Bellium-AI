"""k-NN that recommends an integrator for a normal field. It never integrates.

The features come from the slopes, so a caller can ask before spending the time:
the measured table shows a least-squares solve is exact on a constant slope and
worthless on a dome, at roughly two hundred times the cost of the alternatives.
The published deterministic rule is returned beside the recommendation, and the
memory carries the measured errors that produced each label.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from bellium.material.integration import METHODS, Normals, gradients
from bellium.resources import model_path

SPECIALIST_ID = "bellium/knn/integration-method:v0"
BASELINE_ID = "bellium/deterministic/integration-rule:v0"
SCHEMA = "bellium.integration-method-memory/v1"
FEATURE_NAMES = (
    "direction_focus",
    "slope_uniformity",
    "divergence_ratio",
    "radial_alignment",
    "flat_field",
)
# A tie means the caller should pay less: equal errors are broken by the measured
# p50 latency of the benchmark, cumulative-vertical 1.14 ms, cumulative-average
# 1.17 ms, cumulative 1.20 ms and least-squares 242.2 ms. The same order defines
# the label of a memory item and of a scored field, so the two stay comparable.
TIE_BREAK = ("cumulative-vertical", "cumulative-average", "cumulative", "least-squares")
MIN_POINTS = 4
NEIGHBOURS = 3
MIN_NEIGHBORS = 3
MIN_SIMILARITY = 0.72
# Neighbours vote with the inverse of their distance, so the closest exemplar
# leads without silencing the next ones; the floor keeps an exact match finite.
DISTANCE_FLOOR = 0.001
FLAT_LIMIT = 0.5
FOCUS_LIMIT = 0.6
DIVERGENCE_LIMIT = 0.2
DEFAULT_MEMORY = model_path("knn", "integration", "method-selection-v0.json")


def _unit(name: str, value: object) -> float:
    if value is None:
        raise ValueError(f"feature {name} is unknown; do not coerce to 0")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"feature {name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"feature {name} must be between 0 and 1")
    return number


def _features(source: object) -> dict[str, float]:
    if not isinstance(source, dict):
        raise ValueError("features must be an object")
    return {
        name: _unit(name, source[name] if name in source else None)
        for name in FEATURE_NAMES
    }


def slope_features(slope_x: list[list[float]], slope_y: list[list[float]],
                   valid: list[list[float]]) -> dict[str, float] | None:
    """Five deterministic features of a slope field, or None if it is too small.

    direction_focus is one when every slope points the same way and near zero when
    they spread over the circle; slope_uniformity drops as the gradient magnitude
    varies; divergence_ratio separates a flat ramp from a curved or oscillating
    one; radial_alignment is one when the slopes point away from the centre of the
    valid region, which is what a dome or a cone does; flat_field is one when there
    is no measurable slope at all.
    """
    rows = len(valid)
    columns = len(valid[0])
    points = [
        (r, c) for r in range(rows) for c in range(columns) if valid[r][c]
    ]
    if len(points) < MIN_POINTS:
        return None
    total = len(points)
    magnitudes: list[float] = []
    sum_x = 0.0
    sum_y = 0.0
    for r, c in points:
        gradient_x = slope_x[r][c]
        gradient_y = slope_y[r][c]
        magnitude = math.hypot(gradient_x, gradient_y)
        magnitudes.append(magnitude)
        if magnitude > 1e-12:
            sum_x += gradient_x / magnitude
            sum_y += gradient_y / magnitude
    mean_magnitude = sum(magnitudes) / total
    if mean_magnitude <= 1e-12:
        uniformity = 1.0
        flat = 1.0
    else:
        variance = sum((value - mean_magnitude) ** 2 for value in magnitudes) / total
        uniformity = 1.0 - min(1.0, math.sqrt(variance) / mean_magnitude)
        flat = 0.0
    divergence = 0.0
    for r, c in points:
        value = 0.0
        if c + 1 < columns and valid[r][c + 1]:
            value += slope_x[r][c + 1] - slope_x[r][c]
        if r + 1 < rows and valid[r + 1][c]:
            value += slope_y[r + 1][c] - slope_y[r][c]
        divergence += abs(value)
    divergence_ratio = min(1.0, (divergence / total) / (mean_magnitude + 1e-9))
    center_r = sum(r for r, _ in points) / total
    center_c = sum(c for _, c in points) / total
    radial = 0.0
    for r, c in points:
        gradient_x = slope_x[r][c]
        gradient_y = slope_y[r][c]
        magnitude = math.hypot(gradient_x, gradient_y)
        offset_r = r - center_r
        offset_c = c - center_c
        distance = math.hypot(offset_r, offset_c)
        if magnitude > 1e-12 and distance > 1e-9:
            radial += (gradient_x / magnitude) * (offset_c / distance) + (
                gradient_y / magnitude
            ) * (offset_r / distance)
    return {
        "direction_focus": round(min(1.0, math.hypot(sum_x / total, sum_y / total)), 6),
        "slope_uniformity": round(max(0.0, uniformity), 6),
        "divergence_ratio": round(divergence_ratio, 6),
        "radial_alignment": round(min(1.0, abs(radial / total)), 6),
        "flat_field": flat,
    }


def integration_features(normals: Normals, mask: list[list[float]] | None = None,
                         *, min_cosine: float = 0.0) -> dict[str, float] | None:
    """The same features, straight from a normal field."""
    slope_x, slope_y, valid = gradients(normals, mask, min_cosine=min_cosine)
    return slope_features(slope_x, slope_y, valid)


def deterministic_method(features: dict[str, float]) -> str:
    """The published rule, kept beside the learned tier as the baseline.

    A field that is flat, or that points one way without diverging, has a constant
    slope, and the vertical alignment reproduces a constant slope exactly at about
    one millisecond where the least-squares solve needs two hundred.
    """
    if features["flat_field"] > FLAT_LIMIT:
        return "cumulative-vertical"
    if features["direction_focus"] > FOCUS_LIMIT and (
        features["divergence_ratio"] < DIVERGENCE_LIMIT
    ):
        return "cumulative-vertical"
    return "cumulative-average"


def measured_label(errors: dict[str, float]) -> str:
    """The label a measurement produces: the lowest error, ties by declared cost."""
    missing = sorted(set(METHODS) - set(errors))
    if missing:
        raise ValueError(f"errors must cover every method; missing: {', '.join(missing)}")
    for name, value in errors.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"error for {name} must be numeric")
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"error for {name} must be finite and not negative")
    return min(TIE_BREAK, key=lambda name: errors[name])


def load_memory(path: str | Path | None = None) -> dict[str, Any]:
    """The declared exemplars, validated before they are used."""
    location = Path(path) if path is not None else DEFAULT_MEMORY
    payload = json.loads(location.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise ValueError(f"memory must declare the schema {SCHEMA}")
    items = payload.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("memory must hold at least one item")
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("every memory item must be an object")
        if item.get("method") not in METHODS:
            raise ValueError(f"method must be one of: {', '.join(METHODS)}")
        _features(item.get("features"))
    return payload


def feature_distance(left: dict[str, float], right: dict[str, float]) -> float:
    return math.sqrt(sum((left[name] - right[name]) ** 2 for name in FEATURE_NAMES))


def similarity(left: dict[str, float], right: dict[str, float]) -> float:
    return max(0.0, 1.0 - feature_distance(left, right) / math.sqrt(len(FEATURE_NAMES)))


def recommend_method(
    normals: Normals,
    mask: list[list[float]] | None = None,
    *,
    min_cosine: float = 0.0,
    memory: dict[str, Any] | None = None,
    neighbours: int = NEIGHBOURS,
) -> dict[str, Any]:
    """Recommend one of the four integrators, with the rule's own answer beside it."""
    if isinstance(neighbours, bool) or not isinstance(neighbours, int) or neighbours < 1:
        raise ValueError("neighbours must be a positive integer")
    features = integration_features(normals, mask, min_cosine=min_cosine)
    if features is None:
        return {
            "status": "abstain",
            "reason": "field_too_small",
            "method": None,
            "baseline_method": None,
            "agrees_with_baseline": None,
            "features": None,
            "neighbours": [],
        }
    baseline = deterministic_method(features)
    payload = memory if memory is not None else load_memory()
    scored = sorted(
        (
            (feature_distance(features, _features(item["features"])), item)
            for item in payload["items"]
        ),
        key=lambda pair: pair[0],
    )
    support = [
        pair
        for pair in scored
        if 1.0 - pair[0] / math.sqrt(len(FEATURE_NAMES)) >= MIN_SIMILARITY
    ][:neighbours]
    if len(support) < MIN_NEIGHBORS:
        return {
            "status": "baseline",
            "reason": "not_enough_neighbours",
            "method": baseline,
            "baseline_method": baseline,
            "agrees_with_baseline": True,
            "features": features,
            "neighbours": [
                {
                    "id": item["id"],
                    "method": item["method"],
                    "similarity": round(max(0.0, 1.0 - distance / math.sqrt(len(FEATURE_NAMES))), 6),
                    "errors": item.get("errors", {}),
                }
                for distance, item in support
            ],
        }
    votes: dict[str, float] = {}
    for distance, item in support:
        votes[item["method"]] = votes.get(item["method"], 0.0) + 1.0 / (
            distance + DISTANCE_FLOOR
        )
    winner = max(sorted(votes), key=lambda name: votes[name])
    total = sum(votes.values())
    return {
        "status": "ready",
        "reason": None,
        "method": winner,
        "baseline_method": baseline,
        "agrees_with_baseline": winner == baseline,
        "agreement": round(votes[winner] / total, 6) if total else 0.0,
        "features": features,
        "neighbours": [
            {
                "id": item["id"],
                "method": item["method"],
                "similarity": round(max(0.0, 1.0 - distance / math.sqrt(len(FEATURE_NAMES))), 6),
                "errors": item.get("errors", {}),
            }
            for distance, item in support
        ],
    }
