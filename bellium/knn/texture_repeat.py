"""k-NN that labels a texture repeated or not, and reports measured periods.

The repeat period comes from the deterministic autocorrelation baseline. The
k-NN only labels the family-local verdict; it never replaces a measurement and
never rewrites a pixel. UV repeat counts require an explicit target extent.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from bellium.contracts.schema import AuthorityMode, SpecialistResult
from bellium.knn._texture import local_period_variation, period_estimates, texture_features
from bellium.resources import model_path

SPECIALIST_ID = "bellium/knn/texture-repeat-xy:v0"
SCHEMA = "bellium.texture-repeat-memory/v1"
FAMILIES = ("texture", "tilemap", "sprite")
VERDICTS = ("periodic", "nonperiodic")
FEATURE_NAMES = (
    "strength_x",
    "strength_y",
    "axis_balance",
    "gradient_energy",
    "profile_variance",
    "detail_ratio",
)
MIN_NEIGHBORS = 3
MIN_SIMILARITY = 0.72
DEFAULT_MEMORY = model_path("knn", "visual", "texture-repeat-v0.json")
STRONG_PERIOD = 0.5


def _unit(name: str, value: object) -> float:
    if value is None:
        raise ValueError(f"feature {name} is unknown; do not coerce to 0")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"feature {name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"feature {name} must be between 0 and 1")
    return number


def _features(source: dict[str, Any]) -> dict[str, float]:
    if not isinstance(source, dict):
        raise ValueError("features must be an object")
    return {name: _unit(name, source[name] if name in source else None) for name in FEATURE_NAMES}


def _similarity(left: dict[str, float], right: dict[str, float]) -> float:
    distance = math.sqrt(sum((left[name] - right[name]) ** 2 for name in FEATURE_NAMES))
    return max(0.0, 1.0 - distance / math.sqrt(len(FEATURE_NAMES)))


def _period(raw: object, name: str) -> int | None:
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 2:
        raise ValueError(f"{name} must be an integer period of at least two pixels")
    return raw


def load_repeats(path: str | Path | None = None) -> list[dict[str, Any]]:
    payload = json.loads(Path(path or DEFAULT_MEMORY).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise ValueError("texture repeat memory schema mismatch")
    items = payload.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("texture repeat memory needs items")
    loaded = []
    for raw in items:
        if not isinstance(raw, dict):
            raise ValueError("repeat item must be an object")
        if raw.get("raw_image_stored") is True:
            raise ValueError("raw images must not be stored")
        item_id = str(raw.get("id") or "").strip()
        family = str(raw.get("family") or "").strip()
        source = str(raw.get("source") or "").strip()
        verdict = str(raw.get("verdict") or "").strip()
        if family not in FAMILIES:
            raise ValueError(f"family must be one of: {', '.join(FAMILIES)}")
        if verdict not in VERDICTS:
            raise ValueError("verdict must be periodic or nonperiodic")
        if not item_id or not source:
            raise ValueError("repeat item needs id and source")
        period_x = _period(raw.get("period_x"), "period_x")
        period_y = _period(raw.get("period_y"), "period_y")
        if verdict == "periodic" and period_x is None and period_y is None:
            raise ValueError("a periodic item needs at least one declared period")
        if verdict == "nonperiodic" and (period_x is not None or period_y is not None):
            raise ValueError("a nonperiodic item must not declare a period")
        loaded.append({
            "id": item_id,
            "family": family,
            "verdict": verdict,
            "period_x": period_x,
            "period_y": period_y,
            "source": source,
            "features": _features(raw.get("features") or {}),
        })
    return loaded


def _extent(raw: object) -> tuple[float, float] | None:
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError("target_extent_px must list width and height")
    values = []
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("target_extent_px values must be numeric")
        number = float(value)
        if not math.isfinite(number) or number <= 0:
            raise ValueError("target_extent_px values must be positive")
        values.append(number)
    return values[0], values[1]


def classify_repeat(
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
             "verdict": None, "period_x": None, "period_y": None, "neighbors": []},
            0.0, True, AuthorityMode.CONSULTATIVE,
            notes=("Unknown texture family; no repeat is invented.",),
        )
    image = query.get("image")
    if image is not None:
        features = _features(texture_features(image))
        measured = period_estimates(image)
        variation = None
        for axis, angle in (("x", 0.0), ("y", 90.0)):
            if measured.get(f"period_{axis}") is None:
                continue
            try:
                report = local_period_variation(image, angle)
            except ValueError:
                continue
            if report.get("varies"):
                variation = {"axis": axis, **report}
                break
        if variation is not None:
            return SpecialistResult(
                SPECIALIST_ID,
                {"status": "abstain", "reason": "repeat_varies_across_image", "family": family,
                 "verdict": None, "period_x": None, "period_y": None,
                 "neighbors": [], "variation": variation},
                0.0, True, AuthorityMode.CONSULTATIVE,
                notes=(
                    "Perspective or a gradient: the period changes across the image, so no "
                    "single global period is reported.",
                ),
            )
    else:
        features = _features(query.get("features") or {})
        measured = {
            "period_x": query.get("period_x"),
            "period_y": query.get("period_y"),
            "strength_x": features["strength_x"],
            "strength_y": features["strength_y"],
        }
    records = [item for item in (memory if memory is not None else load_repeats())
               if item["family"] == family]
    scored = sorted(
        ((_similarity(features, item["features"]), item) for item in records),
        key=lambda pair: pair[0], reverse=True,
    )
    nearest = [pair for pair in scored if pair[0] >= MIN_SIMILARITY][:k]
    neighbors = [
        {"id": item["id"], "verdict": item["verdict"], "period_x": item["period_x"],
         "period_y": item["period_y"], "similarity": round(sim, 4), "source": item["source"]}
        for sim, item in nearest
    ]
    if len(nearest) < MIN_NEIGHBORS:
        return SpecialistResult(
            SPECIALIST_ID,
            {"status": "abstain", "reason": "too_few_similar_textures", "family": family,
             "verdict": None, "period_x": None, "period_y": None, "neighbors": neighbors},
            0.0, True, AuthorityMode.CONSULTATIVE,
        )
    verdicts = {item["verdict"] for _, item in nearest}
    if len(verdicts) > 1:
        return SpecialistResult(
            SPECIALIST_ID,
            {"status": "abstain", "reason": "mixed_neighbor_verdicts", "family": family,
             "verdict": None, "period_x": None, "period_y": None, "neighbors": neighbors},
            0.0, True, AuthorityMode.CONSULTATIVE,
            notes=("Disagreeing neighbors do not elect a repeat.",),
        )
    verdict = next(iter(verdicts))
    period_x = measured.get("period_x")
    period_y = measured.get("period_y")
    strong = (measured.get("strength_x", 0.0) >= STRONG_PERIOD,
              measured.get("strength_y", 0.0) >= STRONG_PERIOD)
    if verdict == "periodic" and period_x is None and period_y is None:
        return SpecialistResult(
            SPECIALIST_ID,
            {"status": "abstain", "reason": "period_not_confirmed", "family": family,
             "verdict": None, "period_x": None, "period_y": None, "neighbors": neighbors},
            0.0, True, AuthorityMode.CONSULTATIVE,
            notes=("Neighbors look periodic but no measured period confirmed it.",),
        )
    if verdict == "nonperiodic" and all(strong) and period_x and period_y:
        return SpecialistResult(
            SPECIALIST_ID,
            {"status": "abstain", "reason": "signals_disagree", "family": family,
             "verdict": None, "period_x": None, "period_y": None, "neighbors": neighbors},
            0.0, True, AuthorityMode.CONSULTATIVE,
            notes=("Measured periods contradict the neighbors; abstaining.",),
        )
    extent = _extent(query.get("target_extent_px"))
    repeat_counts = None
    if extent is not None and (period_x or period_y):
        # Only the axes with a measured period produce a count.
        counts: dict[str, Any] = {"target_extent_px": [extent[0], extent[1]]}
        if period_x:
            counts["x"] = int(round(extent[0] / period_x))
        if period_y:
            counts["y"] = int(round(extent[1] / period_y))
        repeat_counts = counts
    strength = sum(sim for sim, _ in nearest) / len(nearest)
    return SpecialistResult(
        SPECIALIST_ID,
        {
            "status": "suggest",
            "family": family,
            "verdict": verdict,
            "period_x": period_x,
            "period_y": period_y,
            "strength_x": measured.get("strength_x"),
            "strength_y": measured.get("strength_y"),
            "repeat_counts": repeat_counts,
            "neighbors": neighbors,
            "pixels_changed": 0,
        },
        round(min(0.9, strength), 4),
        False, AuthorityMode.CONSULTATIVE,
        notes=("Measured periods only. UV repeat counts need a declared extent.",),
    )
