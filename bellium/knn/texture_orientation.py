"""k-NN that names the repeat direction of a texture. It never rewrites pixels.

The direction, the period and the strength come from the deterministic
projection scan. The family-local memory only confirms the grain class, and a
repeat that changes across the image (perspective or a gradient) abstains
instead of being reported as one global period.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from bellium.contracts.schema import AuthorityMode, SpecialistResult
from bellium.knn._texture import (
    direction_scan,
    local_period_variation,
    shift_period,
)
from bellium.knn._image import shape
from bellium.resources import model_path

SPECIALIST_ID = "bellium/knn/texture-orientation:v0"
BASELINE_ID = "bellium/deterministic/grain-axis:v0"
SCHEMA = "bellium.texture-orientation-memory/v1"
FAMILIES = ("texture", "tilemap", "sprite")
GRAINS = ("repeat-x", "repeat-y", "repeat-both", "repeat-diagonal", "no-repeat")
COARSE_ANGLES = tuple(float(angle) for angle in range(0, 180, 15))
REFINE_SPAN_DEG = 10.0
REFINE_STEP_DEG = 2.0
FEATURE_NAMES = (
    "strength_x",
    "strength_y",
    "axis_isotropy",
    "diagonal_strength",
    "diagonal_cos2theta",
    "diagonal_sin2theta",
    "smallest_period_ratio",
)
MIN_NEIGHBORS = 3
MIN_SIMILARITY = 0.72
VARIES_RATIO = 1.15
DEFAULT_MEMORY = model_path("knn", "visual", "texture-orientation-v0.json")


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
    return {name: _unit(name, source[name] if name in source else None) for name in FEATURE_NAMES}


def _similarity(left: dict[str, float], right: dict[str, float]) -> float:
    distance = math.sqrt(sum((left[name] - right[name]) ** 2 for name in FEATURE_NAMES))
    return max(0.0, 1.0 - distance / math.sqrt(len(FEATURE_NAMES)))


def deterministic_grain_verdict(scan: dict[str, Any]) -> str:
    """Published baseline: the grain class implied by the measured directions.

    Axes come first, because a checkerboard is also periodic on the diagonal and
    must not be reported as a diagonal grain.
    """
    if scan.get("axis_x") and scan.get("axis_y"):
        return "repeat-both"
    if scan.get("axis_x"):
        return "repeat-x"
    if scan.get("axis_y"):
        return "repeat-y"
    if scan.get("diagonal"):
        return "repeat-diagonal"
    return "no-repeat"


def refine_angle(image: Any, coarse: float, coarse_period: int | None = None) -> dict[str, Any]:
    """Local search around the coarse angle, because the scan step is 15 degrees.

    The refinement looks for a better angle for the period already measured, so a
    neighbouring angle that accidentally dips at a shorter lag cannot win.
    """
    best = None
    offset = -REFINE_SPAN_DEG
    while offset <= REFINE_SPAN_DEG:
        angle = (coarse + offset) % 180.0
        period, strength = shift_period(image, angle)
        candidate = {
            "angle_deg": round(angle, 3),
            "period": period,
            "strength": round(min(max(strength, 0.0), 1.0), 6),
        }
        if candidate["period"] is None:
            offset += REFINE_STEP_DEG
            continue
        if coarse_period is not None and candidate["period"] != coarse_period:
            offset += REFINE_STEP_DEG
            continue
        if best is None or (candidate["period"], -candidate["strength"]) < (
            best["period"], -best["strength"]
        ):
            best = candidate
        offset += REFINE_STEP_DEG
    return best or {"angle_deg": coarse, "period": None, "strength": 0.0}


def orientation_features(image: Any, scan: dict[str, Any], grain: str) -> dict[str, float]:
    """Signature of a grain class: diagonal fields stay empty unless it is diagonal."""
    height, width = shape(image)
    axis_x = scan.get("axis_x") or {"strength": 0.0, "period": None, "angle_deg": 0.0}
    axis_y = scan.get("axis_y") or {"strength": 0.0, "period": None, "angle_deg": 90.0}
    diagonal = scan.get("diagonal") if grain == "repeat-diagonal" else None
    best = scan.get("best")
    strength_x = float(axis_x["strength"])
    strength_y = float(axis_y["strength"])
    smaller = min(strength_x, strength_y)
    larger = max(strength_x, strength_y)
    limit = 0.5 * min(width, height)
    diagonal_angle = float(diagonal["angle_deg"]) if diagonal else 0.0
    smallest = float(best["period"]) if best and best.get("period") else 0.0
    return {
        "strength_x": round(min(max(strength_x, 0.0), 1.0), 6),
        "strength_y": round(min(max(strength_y, 0.0), 1.0), 6),
        "axis_isotropy": round(0.0 if larger == 0.0 else smaller / larger, 6),
        "diagonal_strength": round(
            0.0 if diagonal is None else min(max(float(diagonal["strength"]), 0.0), 1.0), 6
        ),
        "diagonal_cos2theta": round((math.cos(2 * math.radians(diagonal_angle)) + 1.0) / 2.0, 6),
        "diagonal_sin2theta": round((math.sin(2 * math.radians(diagonal_angle)) + 1.0) / 2.0, 6),
        "smallest_period_ratio": round(0.0 if limit == 0 else min(smallest / limit, 1.0), 6),
    }


def anchor_entry(scan: dict[str, Any], grain: str) -> dict[str, Any] | None:
    """The measured direction the reported angle comes from, per grain class."""
    if grain == "repeat-x":
        return scan.get("axis_x")
    if grain == "repeat-y":
        return scan.get("axis_y")
    if grain == "repeat-both":
        return scan.get("axis_x") or scan.get("axis_y")
    if grain == "repeat-diagonal":
        return scan.get("diagonal")
    return None


def load_orientations(path: str | Path | None = None) -> list[dict[str, Any]]:
    payload = json.loads(Path(path or DEFAULT_MEMORY).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise ValueError("texture orientation memory schema mismatch")
    items = payload.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("texture orientation memory needs items")
    loaded = []
    for raw in items:
        if not isinstance(raw, dict):
            raise ValueError("orientation item must be an object")
        if raw.get("raw_image_stored") is True:
            raise ValueError("raw images must not be stored")
        item_id = str(raw.get("id") or "").strip()
        family = str(raw.get("family") or "").strip()
        grain = str(raw.get("grain") or "").strip()
        source = str(raw.get("source") or "").strip()
        if family not in FAMILIES:
            raise ValueError(f"family must be one of: {', '.join(FAMILIES)}")
        if grain not in GRAINS:
            raise ValueError(f"grain must be one of: {', '.join(GRAINS)}")
        if not item_id or not source:
            raise ValueError("orientation item needs id and source")
        loaded.append({
            "id": item_id,
            "family": family,
            "grain": grain,
            "source": source,
            "features": _features(raw.get("features")),
        })
    return loaded


def classify_orientation(
    query: dict[str, Any],
    *,
    memory: list[dict[str, Any]] | None = None,
    k: int = 5,
) -> SpecialistResult:
    family = str(query.get("family") or "").strip()
    if family not in FAMILIES:
        return SpecialistResult(
            SPECIALIST_ID,
            {"status": "abstain", "reason": "unknown_family", "family": family,
             "grain": None, "neighbors": []},
            0.0, True, AuthorityMode.CONSULTATIVE,
            notes=("Unknown texture family; no grain is invented.",),
        )
    image = query.get("image")
    if image is not None:
        scan = direction_scan(image, COARSE_ANGLES)
        baseline = deterministic_grain_verdict(scan)
        anchor = anchor_entry(scan, baseline)
        features = _features(orientation_features(image, scan, baseline))
        if anchor is None:
            refined = {"angle_deg": None, "period": None, "strength": 0.0}
            variation = {"varies": False, "ratio": None, "reason": "no_direction"}
        elif baseline in ("repeat-x", "repeat-y", "repeat-both"):
            # Axis classes report their axis; only a diagonal needs a refined angle.
            refined = {"angle_deg": float(anchor["angle_deg"]),
                       "period": anchor["period"], "strength": anchor["strength"]}
            variation = local_period_variation(image, float(anchor["angle_deg"]))
        else:
            refined = refine_angle(image, float(anchor["angle_deg"]), anchor["period"])
            variation = local_period_variation(image, refined["angle_deg"])
    else:
        scan = query.get("scan")
        if not isinstance(scan, dict) or "best" not in scan or "angles" not in scan:
            raise ValueError("query needs an image or a complete scan object")
        baseline = deterministic_grain_verdict(scan)
        anchor = anchor_entry(scan, baseline)
        features = _features(query.get("features"))
        if anchor is None:
            refined = {"angle_deg": None, "period": None, "strength": 0.0}
        else:
            refined = {"angle_deg": float(anchor["angle_deg"]),
                       "period": anchor["period"], "strength": anchor["strength"]}
        variation = query.get("variation") or {"varies": False, "ratio": None, "reason": "not_measured"}
    records = [item for item in (memory if memory is not None else load_orientations())
               if item["family"] == family]
    scored = sorted(
        ((_similarity(features, item["features"]), item) for item in records),
        key=lambda pair: pair[0], reverse=True,
    )
    nearest = [pair for pair in scored if pair[0] >= MIN_SIMILARITY][:k]
    neighbors = [
        {"id": item["id"], "grain": item["grain"], "similarity": round(sim, 4),
         "source": item["source"]}
        for sim, item in nearest
    ]
    output: dict[str, Any] = {
        "family": family,
        "baseline": baseline,
        "baseline_id": BASELINE_ID,
        "angle_deg": refined["angle_deg"],
        "period": refined["period"],
        "strength": refined["strength"],
        "axis_periods": {
            "x": (scan.get("axis_x") or {}).get("period"),
            "y": (scan.get("axis_y") or {}).get("period"),
        },
        "features": features,
        "neighbors": neighbors,
        "local_variation": variation,
        "pixels_changed": 0,
    }
    if variation.get("varies"):
        output.update(status="abstain", reason="repeat_varies_across_image", grain=None)
        return SpecialistResult(
            SPECIALIST_ID, output, 0.0, True, AuthorityMode.CONSULTATIVE,
            notes=("Perspective or a gradient: no single global period is reported.",),
        )
    if len(nearest) < MIN_NEIGHBORS:
        output.update(status="abstain", reason="too_few_similar_textures", grain=None)
        return SpecialistResult(
            SPECIALIST_ID, output, 0.0, True, AuthorityMode.CONSULTATIVE,
        )
    grains = {item["grain"] for _, item in nearest}
    if len(grains) > 1:
        output.update(status="abstain", reason="mixed_neighbor_grains", grain=None)
        return SpecialistResult(
            SPECIALIST_ID, output, 0.0, True, AuthorityMode.CONSULTATIVE,
            notes=("Disagreeing neighbours do not elect a grain.",),
        )
    grain = next(iter(grains))
    if grain != baseline:
        output.update(status="abstain", reason="baseline_disagrees", grain=None)
        return SpecialistResult(
            SPECIALIST_ID, output, 0.0, True, AuthorityMode.CONSULTATIVE,
            notes=("The memory and the published axis rule disagree.",),
        )
    strength = sum(sim for sim, _ in nearest) / len(nearest)
    output.update(status="suggest", grain=grain)
    return SpecialistResult(
        SPECIALIST_ID, output, round(min(0.9, strength), 4), False, AuthorityMode.CONSULTATIVE,
        notes=(
            "Direction and period are measured, the grain class is confirmed by the memory.",
            "Angles are resolved at two degrees after a fifteen-degree coarse scan.",
        ),
    )
