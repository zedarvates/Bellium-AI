"""k-NN that says whether an animation clip closes its loop. No pixels are stored.

The reference set is the known clean loops of one family. A pair inside that
space is a loop, a pair clearly outside it drifts, and the band between the two
abstains. The published threshold baseline must agree in both decided cases.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from bellium.contracts.schema import AuthorityMode, SpecialistResult
from bellium.knn._image import Mask, validate_mask
from bellium.knn._sheet import (
    deterministic_loop_verdict,
    loop_features,
)
from bellium.resources import model_path

SPECIALIST_ID = "bellium/knn/clip-loop:v0"
BASELINE_ID = "bellium/deterministic/clip-loop-threshold:v0"
SCHEMA = "bellium.clip-loop-memory/v1"
FAMILIES = ("character", "effect", "ui", "prop")
VERDICTS = ("loops", "drifts")
FEATURE_NAMES = (
    "iou_mismatch",
    "added_ratio",
    "removed_ratio",
    "centroid_shift",
    "area_ratio",
    "color_delta",
    "bbox_shift",
)
MIN_NEIGHBORS = 3
INSIDE_SIMILARITY = 0.95
OUTSIDE_SIMILARITY = 0.7
DEFAULT_MEMORY = model_path("knn", "layout", "clip-loop-v0.json")


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


def _frame(raw: object, name: str) -> tuple[Mask, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"{name} frame must be an object with image and mask")
    image = raw.get("image")
    mask = raw.get("mask")
    if image is None or mask is None:
        raise ValueError(f"{name} frame needs image and mask")
    validate_mask(image, mask)
    return mask, image


def load_clips(path: str | Path | None = None) -> list[dict[str, Any]]:
    payload = json.loads(Path(path or DEFAULT_MEMORY).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise ValueError("clip loop memory schema mismatch")
    items = payload.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("clip loop memory needs items")
    loaded = []
    for raw in items:
        if not isinstance(raw, dict):
            raise ValueError("clip item must be an object")
        if raw.get("raw_image_stored") is True:
            raise ValueError("raw images must not be stored")
        item_id = str(raw.get("id") or "").strip()
        family = str(raw.get("family") or "").strip()
        source = str(raw.get("source") or "").strip()
        verdict = str(raw.get("verdict") or "").strip()
        if family not in FAMILIES:
            raise ValueError(f"family must be one of: {', '.join(FAMILIES)}")
        if verdict not in VERDICTS:
            raise ValueError("verdict must be loops or drifts")
        if not item_id or not source:
            raise ValueError("clip item needs id and source")
        loaded.append({
            "id": item_id,
            "family": family,
            "verdict": verdict,
            "source": source,
            "features": _features(raw.get("features")),
        })
    return loaded


def classify_loop(
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
             "verdict": None, "neighbors": []},
            0.0, True, AuthorityMode.CONSULTATIVE,
            notes=("Unknown clip family; no loop verdict is invented.",),
        )
    if query.get("frames") is not None:
        frames = query["frames"]
        if not isinstance(frames, dict):
            raise ValueError("frames must be an object with first and last")
        features = _features(loop_features(_frame(frames.get("first"), "first"),
                                           _frame(frames.get("last"), "last")))
    else:
        features = _features(query.get("features"))
    baseline = deterministic_loop_verdict(features)
    records = [item for item in (memory if memory is not None else load_clips())
               if item["family"] == family]
    scored = sorted(
        ((_similarity(features, item["features"]), item) for item in records),
        key=lambda pair: pair[0], reverse=True,
    )
    nearest = scored[:k]
    neighbors = [
        {"id": item["id"], "verdict": item["verdict"], "similarity": round(sim, 4),
         "source": item["source"]}
        for sim, item in nearest
    ]
    reference = [pair for pair in scored if pair[1]["verdict"] == "loops"][:k]
    output = {
        "family": family,
        "baseline": baseline,
        "baseline_id": BASELINE_ID,
        "features": features,
        "neighbors": neighbors,
        "reference_count": len(reference),
        "pixels_changed": 0,
    }
    if len(reference) < MIN_NEIGHBORS:
        output.update(status="abstain", reason="too_few_known_loops", verdict=None)
        return SpecialistResult(
            SPECIALIST_ID, output, 0.0, True, AuthorityMode.CONSULTATIVE,
        )
    closeness = sum(sim for sim, _ in reference) / len(reference)
    output["loop_space_similarity"] = round(closeness, 4)
    if closeness >= INSIDE_SIMILARITY:
        verdict = "loops"
    elif closeness < OUTSIDE_SIMILARITY and baseline == "drifts":
        verdict = "drifts"
    else:
        output.update(status="abstain", reason="between_known_spaces", verdict=None)
        return SpecialistResult(
            SPECIALIST_ID, output, 0.0, True, AuthorityMode.CONSULTATIVE,
            notes=("Neither a known clean loop nor clearly outside the loop space.",),
        )
    if verdict != baseline:
        output.update(status="abstain", reason="baseline_disagrees", verdict=None)
        return SpecialistResult(
            SPECIALIST_ID, output, 0.0, True, AuthorityMode.CONSULTATIVE,
            notes=("The memory and the published threshold disagree.",),
        )
    output.update(status="suggest", verdict=verdict)
    return SpecialistResult(
        SPECIALIST_ID, output, round(min(0.9, closeness), 4), False, AuthorityMode.CONSULTATIVE,
        notes=("Geometric loop check only. It neither retimes nor exports the clip.",),
    )
