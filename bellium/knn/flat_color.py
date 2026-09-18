"""k-NN that labels a region flat or textured. Does not paint."""

from __future__ import annotations

import json
import math
from typing import Any

from bellium.contracts.schema import AuthorityMode, SpecialistResult
from bellium.knn._image import Image, Mask, shape, validate_mask
from bellium.resources import model_path

SPECIALIST_ID = "bellium/knn/flat-color:v0"
SCHEMA = "bellium.flat-color-memory/v1"
FEATURE_NAMES = (
    "luma_std",
    "chroma",
    "chroma_std",
    "unique_ratio",
    "channel_span",
)
MIN_NEIGHBORS = 3
MIN_SIMILARITY = 0.70
DEFAULT_MEMORY = model_path("knn", "visual", "flat-color-v0.json")


def _luma(pixel: tuple[int, int, int]) -> float:
    return (0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.0722 * pixel[2]) / 255.0


def extract_features(image: Image, mask: Mask | None = None) -> dict[str, float]:
    height, width = shape(image)
    if mask is not None:
        validate_mask(image, mask)
        pixels = [image[r][c] for r in range(height) for c in range(width) if not mask[r][c]]
    else:
        pixels = [pixel for row in image for pixel in row]
    if len(pixels) < 4:
        raise ValueError("need at least four unmasked pixels")
    count = len(pixels)
    lumas = [_luma(pixel) for pixel in pixels]
    mean_l = sum(lumas) / count
    var_l = sum((value - mean_l) ** 2 for value in lumas) / count
    chromas = [(max(p) - min(p)) / 255.0 for p in pixels]
    mean_c = sum(chromas) / count
    var_c = sum((value - mean_c) ** 2 for value in chromas) / count
    quantized = {(p[0] // 16, p[1] // 16, p[2] // 16) for p in pixels}
    span = max(max(p[i] for p in pixels) - min(p[i] for p in pixels) for i in range(3)) / 255.0
    return {
        "luma_std": min(math.sqrt(var_l) / 0.5, 1.0),
        "chroma": min(mean_c, 1.0),
        "chroma_std": min(math.sqrt(var_c) / 0.5, 1.0),
        "unique_ratio": min(len(quantized) / count, 1.0),
        "channel_span": min(span, 1.0),
    }


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


def load_flats(path=None) -> list[dict[str, Any]]:
    payload = json.loads((path or DEFAULT_MEMORY).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise ValueError("flat-color memory schema mismatch")
    items = payload.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("flat-color memory needs items")
    loaded = []
    for raw in items:
        if not isinstance(raw, dict):
            raise ValueError("item must be an object")
        if raw.get("raw_image_stored") is True:
            raise ValueError("raw images must not be stored")
        item_id = str(raw.get("id") or "").strip()
        label = str(raw.get("label") or "").strip()
        source = str(raw.get("source") or "").strip()
        if label not in {"flat", "textured"}:
            raise ValueError("label must be flat or textured")
        if not item_id or not source:
            raise ValueError("item needs id and source")
        loaded.append({
            "id": item_id,
            "label": label,
            "source": source,
            "features": _features(raw.get("features")),
        })
    return loaded


def classify_flat(
    query: dict[str, Any],
    *,
    memory: list[dict[str, Any]] | None = None,
    k: int = 5,
) -> SpecialistResult:
    if query.get("image") is not None:
        features = extract_features(query["image"], query.get("mask"))
    else:
        features = _features(query.get("features"))
    records = memory if memory is not None else load_flats()
    scored = sorted(
        ((_similarity(features, item["features"]), item) for item in records),
        key=lambda pair: pair[0], reverse=True,
    )
    nearest = [pair for pair in scored if pair[0] >= MIN_SIMILARITY][:k]
    neighbors = [
        {"id": item["id"], "label": item["label"], "similarity": round(sim, 4)}
        for sim, item in nearest
    ]
    if len(nearest) < MIN_NEIGHBORS:
        return SpecialistResult(
            SPECIALIST_ID,
            {"status": "abstain", "reason": "too_few_similar_regions", "label": None,
             "neighbors": neighbors},
            0.0, True, AuthorityMode.CONSULTATIVE,
        )
    labels = {item["label"] for _, item in nearest}
    if len(labels) > 1:
        return SpecialistResult(
            SPECIALIST_ID,
            {"status": "abstain", "reason": "mixed_neighbor_labels", "label": None,
             "neighbors": neighbors},
            0.0, True, AuthorityMode.CONSULTATIVE,
            notes=("Disagreeing neighbors do not elect a fill.",),
        )
    return SpecialistResult(
        SPECIALIST_ID,
        {"status": "suggest", "label": next(iter(labels)), "neighbors": neighbors},
        round(sum(sim for sim, _ in nearest) / len(nearest), 4),
        False, AuthorityMode.CONSULTATIVE,
        notes=("Classification only. Painting is a separate consultative step.",),
    )

