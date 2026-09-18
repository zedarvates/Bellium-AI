"""k-NN that proposes a sprite anchor inside the content box. It never cuts pixels."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from bellium.contracts.schema import AuthorityMode, SpecialistResult
from bellium.knn._image import Mask, validate_mask
from bellium.resources import model_path

SPECIALIST_ID = "bellium/knn/sprite-anchor:v0"
SCHEMA = "bellium.sprite-anchor-memory/v1"
FAMILIES = ("character", "effect", "ui", "prop")
ANCHORS = ("feet", "center", "top-center", "bottom-center")
FEATURE_NAMES = (
    "coverage",
    "bbox_fill",
    "centroid_x",
    "centroid_y",
    "bottom_mass",
    "symmetry",
    "aspect",
)
MIN_NEIGHBORS = 3
MIN_SIMILARITY = 0.7
DEFAULT_MEMORY = model_path("knn", "visual", "sprite-anchor-v0.json")


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


def _bbox(mask: Mask) -> tuple[int, int, int, int] | None:
    rows = [r for r, row in enumerate(mask) if any(row)]
    cols = [c for c in range(len(mask[0])) if any(row[c] for row in mask)]
    if not rows or not cols:
        return None
    return min(cols), min(rows), max(cols), max(rows)


def anchor_features(mask: Mask) -> dict[str, float]:
    """Silhouette features in [0, 1] from a foreground mask."""
    if not isinstance(mask, list) or not mask or not mask[0]:
        raise ValueError("mask is empty")
    width = len(mask[0])
    if any(len(row) != width for row in mask):
        raise ValueError("mask rows must have equal width")
    for row in mask:
        for value in row:
            if value not in (0, 1):
                raise ValueError("mask values must be 0 or 1")
    box = _bbox(mask)
    if box is None:
        raise ValueError("mask has no foreground pixel")
    x0, y0, x1, y1 = box
    box_w = x1 - x0 + 1
    box_h = y1 - y0 + 1
    height = len(mask)
    total = sum(sum(row) for row in mask)
    weighted_x = sum(c for r in range(height) for c in range(width) if mask[r][c])
    weighted_y = sum(r for r in range(height) for c in range(width) if mask[r][c])
    centroid_x = ((weighted_x / total) - x0) / max(box_w - 1, 1)
    centroid_y = ((weighted_y / total) - y0) / max(box_h - 1, 1)
    bottom_rows = range(y0 + (2 * box_h) // 3, y1 + 1)
    bottom = sum(mask[r][c] for r in bottom_rows for c in range(x0, x1 + 1))
    mirrored = 0
    for r in range(height):
        for c in range(width):
            if mask[r][c] and mask[r][width - 1 - c]:
                mirrored += 1
    return {
        "coverage": round(min(total / (width * height), 1.0), 6),
        "bbox_fill": round(min(total / (box_w * box_h), 1.0), 6),
        "centroid_x": round(min(max(centroid_x, 0.0), 1.0), 6),
        "centroid_y": round(min(max(centroid_y, 0.0), 1.0), 6),
        "bottom_mass": round(min(bottom / total, 1.0), 6),
        "symmetry": round(min((2 * mirrored) / total, 1.0), 6),
        "aspect": round(min(box_w, box_h) / max(box_w, box_h), 6),
    }


def load_anchors(path: str | Path | None = None) -> list[dict[str, Any]]:
    payload = json.loads(Path(path or DEFAULT_MEMORY).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise ValueError("sprite anchor memory schema mismatch")
    items = payload.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("sprite anchor memory needs items")
    loaded = []
    for raw in items:
        if not isinstance(raw, dict):
            raise ValueError("anchor item must be an object")
        if raw.get("raw_image_stored") is True:
            raise ValueError("raw images must not be stored")
        item_id = str(raw.get("id") or "").strip()
        family = str(raw.get("family") or "").strip()
        anchor = str(raw.get("anchor") or "").strip()
        source = str(raw.get("source") or "").strip()
        if family not in FAMILIES:
            raise ValueError(f"family must be one of: {', '.join(FAMILIES)}")
        if anchor not in ANCHORS:
            raise ValueError(f"anchor must be one of: {', '.join(ANCHORS)}")
        if not item_id or not source:
            raise ValueError("anchor item needs id and source")
        anchor_x = _unit("anchor_x", raw.get("anchor_x"))
        anchor_y = _unit("anchor_y", raw.get("anchor_y"))
        loaded.append({
            "id": item_id,
            "family": family,
            "anchor": anchor,
            "anchor_x": anchor_x,
            "anchor_y": anchor_y,
            "source": source,
            "features": _features(raw.get("features")),
        })
    return loaded


def propose_anchor(
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
             "anchor": None, "neighbors": []},
            0.0, True, AuthorityMode.CONSULTATIVE,
            notes=("Unknown sprite family; no anchor is invented.",),
        )
    image = query.get("image")
    if image is not None:
        mask = query.get("mask")
        if mask is None:
            raise ValueError("query needs a foreground mask")
        validate_mask(image, mask)
        features = _features(anchor_features(mask))
    else:
        features = _features(query.get("features") or {})
    records = [item for item in (memory if memory is not None else load_anchors())
               if item["family"] == family]
    scored = sorted(
        ((_similarity(features, item["features"]), item) for item in records),
        key=lambda pair: pair[0], reverse=True,
    )
    nearest = [pair for pair in scored if pair[0] >= MIN_SIMILARITY][:k]
    neighbors = [
        {"id": item["id"], "anchor": item["anchor"], "anchor_x": item["anchor_x"],
         "anchor_y": item["anchor_y"], "similarity": round(sim, 4), "source": item["source"]}
        for sim, item in nearest
    ]
    if len(nearest) < MIN_NEIGHBORS:
        return SpecialistResult(
            SPECIALIST_ID,
            {"status": "abstain", "reason": "too_few_similar_sprites", "family": family,
             "anchor": None, "neighbors": neighbors},
            0.0, True, AuthorityMode.CONSULTATIVE,
        )
    anchors = {item["anchor"] for _, item in nearest}
    if len(anchors) > 1:
        return SpecialistResult(
            SPECIALIST_ID,
            {"status": "abstain", "reason": "mixed_neighbor_anchors", "family": family,
             "anchor": None, "neighbors": neighbors},
            0.0, True, AuthorityMode.CONSULTATIVE,
            notes=("Disagreeing neighbors do not elect an anchor.",),
        )
    anchor = next(iter(anchors))
    agreeing = [item for _, item in nearest if item["anchor"] == anchor]
    anchor_x = sum(item["anchor_x"] for item in agreeing) / len(agreeing)
    anchor_y = sum(item["anchor_y"] for item in agreeing) / len(agreeing)
    spread = max(
        max(abs(item["anchor_x"] - anchor_x) for item in agreeing),
        max(abs(item["anchor_y"] - anchor_y) for item in agreeing),
    )
    strength = sum(sim for sim, _ in nearest) / len(nearest)
    return SpecialistResult(
        SPECIALIST_ID,
        {
            "status": "suggest",
            "family": family,
            "anchor": anchor,
            "anchor_x": round(anchor_x, 4),
            "anchor_y": round(anchor_y, 4),
            "spread": round(spread, 4),
            "centroid": {"x": features["centroid_x"], "y": features["centroid_y"]},
            "neighbors": neighbors,
            "pixels_changed": 0,
        },
        round(min(0.9, strength), 4),
        False, AuthorityMode.CONSULTATIVE,
        notes=("Anchor is relative to the content box, not to the frame edge.",),
    )
