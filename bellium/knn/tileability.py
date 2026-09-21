"""k-NN that labels wrap continuity per axis. It does not repair the seam."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from bellium.contracts.schema import AuthorityMode, SpecialistResult
from bellium.knn._texture import (
    SEAM_CHANNELS,
    deterministic_seam_verdict,
    seam_features,
)
from bellium.resources import model_path

SPECIALIST_ID = "bellium/knn/texture-tileability:v0"
SCHEMA = "bellium.tileability-memory/v1"
FAMILIES = ("texture", "tilemap", "sprite")
AXES = ("x", "y")
VERDICTS = ("tileable", "seam")
FEATURE_NAMES = (*SEAM_CHANNELS, "axis_asym")
MIN_NEIGHBORS = 3
MIN_SIMILARITY = 0.7
DEFAULT_MEMORY = model_path("knn", "visual", "tileability-v0.json")


def _unit(name: str, value: object) -> float:
    if value is None:
        raise ValueError(f"feature {name} is unknown; do not coerce to 0")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"feature {name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"feature {name} must be between 0 and 1")
    return number


def axis_features(seams: dict[str, dict[str, float]], axis: str) -> dict[str, float]:
    """Memory features for one axis: its seam channels plus axis asymmetry."""
    if axis not in AXES:
        raise ValueError("axis must be x or y")
    values = {name: _unit(name, seams[axis][name]) for name in SEAM_CHANNELS}
    other = "y" if axis == "x" else "x"
    values["axis_asym"] = round(min(1.0, abs(seams[axis]["gap"] - seams[other]["gap"])), 6)
    return values


def _features(source: object) -> dict[str, float]:
    if not isinstance(source, dict):
        raise ValueError("features must be an object")
    return {name: _unit(name, source[name] if name in source else None) for name in FEATURE_NAMES}


def _similarity(left: dict[str, float], right: dict[str, float]) -> float:
    distance = math.sqrt(sum((left[name] - right[name]) ** 2 for name in FEATURE_NAMES))
    return max(0.0, 1.0 - distance / math.sqrt(len(FEATURE_NAMES)))


def load_tiles(path: str | Path | None = None) -> list[dict[str, Any]]:
    payload = json.loads(Path(path or DEFAULT_MEMORY).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise ValueError("tileability memory schema mismatch")
    items = payload.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("tileability memory needs items")
    loaded = []
    for raw in items:
        if not isinstance(raw, dict):
            raise ValueError("tileability item must be an object")
        if raw.get("raw_image_stored") is True:
            raise ValueError("raw images must not be stored")
        item_id = str(raw.get("id") or "").strip()
        family = str(raw.get("family") or "").strip()
        axis = str(raw.get("axis") or "").strip()
        source = str(raw.get("source") or "").strip()
        verdict = str(raw.get("verdict") or "").strip()
        if family not in FAMILIES:
            raise ValueError(f"family must be one of: {', '.join(FAMILIES)}")
        if axis not in AXES:
            raise ValueError("axis must be x or y")
        if verdict not in VERDICTS:
            raise ValueError("verdict must be tileable or seam")
        if not item_id or not source:
            raise ValueError("tileability item needs id and source")
        loaded.append({
            "id": item_id,
            "family": family,
            "axis": axis,
            "verdict": verdict,
            "source": source,
            "features": _features(raw.get("features")),
        })
    return loaded


def tileability_features(image: Any) -> dict[str, dict[str, float]]:
    """Per-axis memory features measured from an image."""
    seams = seam_features(image)
    return {axis: axis_features(seams, axis) for axis in AXES}


def classify_tileability(
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
             "axes": {}, "neighbors": []},
            0.0, True, AuthorityMode.CONSULTATIVE,
            notes=("Unknown texture family; no verdict is invented.",),
        )
    image = query.get("image")
    if image is not None:
        features = tileability_features(image)
    else:
        raw = query.get("features")
        if not isinstance(raw, dict) or set(raw) != set(AXES):
            raise ValueError("features must provide x and y axis objects")
        features = {axis: _features(raw[axis]) for axis in AXES}
    records = [item for item in (memory if memory is not None else load_tiles())
               if item["family"] == family]
    axes: dict[str, Any] = {}
    neighbors: list[dict[str, Any]] = []
    weakest = 1.0
    for axis in AXES:
        scored = sorted(
            ((_similarity(features[axis], item["features"]), item)
             for item in records if item["axis"] == axis),
            key=lambda pair: pair[0], reverse=True,
        )
        nearest = [pair for pair in scored if pair[0] >= MIN_SIMILARITY][:k]
        axis_neighbors = [
            {"id": item["id"], "axis": axis, "verdict": item["verdict"],
             "similarity": round(sim, 4), "source": item["source"]}
            for sim, item in nearest
        ]
        neighbors.extend(axis_neighbors)
        baseline = deterministic_seam_verdict(features[axis])
        if len(nearest) < MIN_NEIGHBORS:
            axes[axis] = {"status": "abstain", "reason": "too_few_similar_tiles",
                          "verdict": None, "baseline": baseline,
                          "features": features[axis], "neighbors": axis_neighbors}
            weakest = 0.0
            continue
        verdicts = {item["verdict"] for _, item in nearest}
        if len(verdicts) > 1:
            axes[axis] = {"status": "abstain", "reason": "mixed_neighbor_verdicts",
                          "verdict": None, "baseline": baseline,
                          "features": features[axis], "neighbors": axis_neighbors}
            weakest = 0.0
            continue
        verdict = next(iter(verdicts))
        if (verdict == "tileable") != (baseline == "continuous"):
            axes[axis] = {"status": "abstain", "reason": "baseline_disagrees",
                          "verdict": None, "baseline": baseline,
                          "features": features[axis], "neighbors": axis_neighbors}
            weakest = 0.0
            continue
        strength = sum(sim for sim, _ in nearest) / len(nearest)
        weakest = min(weakest, strength)
        axes[axis] = {"status": "suggest", "verdict": verdict, "baseline": baseline,
                      "features": features[axis], "neighbors": axis_neighbors,
                      "confidence": round(strength, 4)}
    decided = [axis for axis in AXES if axes[axis]["status"] == "suggest"]
    if not decided:
        return SpecialistResult(
            SPECIALIST_ID,
            {"status": "abstain", "reason": "no_axis_confirmed", "family": family,
             "axes": axes, "neighbors": neighbors},
            0.0, True, AuthorityMode.CONSULTATIVE,
            notes=("The deterministic seam baseline and the neighbors did not agree.",),
        )
    seams = [axis for axis in decided if axes[axis]["verdict"] == "seam"]
    return SpecialistResult(
        SPECIALIST_ID,
        {
            "status": "suggest",
            "family": family,
            "verdict": "seam" if seams else "tileable",
            "seam_axes": seams,
            "undecided_axes": [axis for axis in AXES if axis not in decided],
            "axes": axes,
            "neighbors": neighbors,
            "pixels_changed": 0,
        },
        round(weakest, 4),
        False, AuthorityMode.CONSULTATIVE,
        notes=("Wrap continuity only. Repairing a seam is a separate bounded step.",),
    )
