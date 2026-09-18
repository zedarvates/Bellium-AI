"""Domain-isolated visual novelty k-NN. Does not store pixels."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from bellium.contracts.schema import AuthorityMode, SpecialistResult
from bellium.resources import model_path
from bellium.knn._image import Image, shape

SPECIALIST_ID = "bellium/knn/visual-anomaly:v0"
SCHEMA = "bellium.visual-anomaly-memory/v1"
FEATURE_NAMES = (
    "mean_r",
    "mean_g",
    "mean_b",
    "luma_std",
    "dark_ratio",
    "border_delta",
    "chroma",
)
LABELS = ("normal", "novel")
MIN_SIZE = 4
MIN_NEIGHBORS = 3
RETRIEVE_SIM = 0.40
IN_SPACE_SIM = 0.88
DEFAULT_MEMORY = (
    model_path()
    / "knn"
    / "visual"
    / "anomaly-exemplars-v0.json"
)


def _luma(pixel: tuple[int, int, int]) -> float:
    return (0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.0722 * pixel[2]) / 255.0


def extract_features(image: Image) -> dict[str, float]:
    height, width = shape(image)
    if height < MIN_SIZE or width < MIN_SIZE:
        raise ValueError(f"image must be at least {MIN_SIZE}x{MIN_SIZE}")
    pixels = [pixel for row in image for pixel in row]
    count = len(pixels)
    mean_r = sum(pixel[0] for pixel in pixels) / count / 255.0
    mean_g = sum(pixel[1] for pixel in pixels) / count / 255.0
    mean_b = sum(pixel[2] for pixel in pixels) / count / 255.0
    lumas = [_luma(pixel) for pixel in pixels]
    luma_mean = sum(lumas) / count
    variance = sum((value - luma_mean) ** 2 for value in lumas) / count
    border = []
    center = []
    r0, r1 = height // 4, height - height // 4
    c0, c1 = width // 4, width - width // 4
    for row in range(height):
        for col in range(width):
            value = lumas[row * width + col]
            if row < r0 or row >= r1 or col < c0 or col >= c1:
                border.append(value)
            else:
                center.append(value)
    border_mean = sum(border) / len(border)
    center_mean = sum(center) / len(center) if center else border_mean
    return {
        "mean_r": mean_r,
        "mean_g": mean_g,
        "mean_b": mean_b,
        "luma_std": min(math.sqrt(variance) / 0.5, 1.0),
        "dark_ratio": sum(1 for value in lumas if value < 0.12) / count,
        "border_delta": min(abs(border_mean - center_mean), 1.0),
        "chroma": max(mean_r, mean_g, mean_b) - min(mean_r, mean_g, mean_b),
    }


def _features_from_query(query: dict[str, Any]) -> dict[str, float]:
    if query.get("image") is not None:
        return extract_features(query["image"])
    source = query.get("features")
    if not isinstance(source, dict):
        raise ValueError("query needs image or features")
    values: dict[str, float] = {}
    for name in FEATURE_NAMES:
        if name not in source:
            raise ValueError(f"missing feature {name}")
        raw = source[name]
        if raw is None:
            raise ValueError(f"feature {name} is unknown; do not coerce to 0")
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(f"feature {name} must be numeric")
        number = float(raw)
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise ValueError(f"feature {name} must be between 0 and 1")
        values[name] = number
    return values


def _similarity(left: dict[str, float], right: dict[str, float]) -> float:
    distance = math.sqrt(
        sum((left[name] - right[name]) ** 2 for name in FEATURE_NAMES)
    )
    return max(0.0, 1.0 - distance / math.sqrt(len(FEATURE_NAMES)))


def load_exemplars(path: str | Path | None = None) -> list[dict[str, Any]]:
    payload = json.loads(Path(path or DEFAULT_MEMORY).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise ValueError("visual anomaly memory schema mismatch")
    items = payload.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("visual anomaly memory needs items")
    loaded = []
    for raw in items:
        if not isinstance(raw, dict):
            raise ValueError("exemplar must be an object")
        if raw.get("raw_image_stored") is True:
            raise ValueError("raw images must not be stored")
        label = str(raw.get("label") or "").strip()
        if label not in LABELS:
            raise ValueError("label must be normal or novel")
        domain = str(raw.get("domain") or "").strip()
        item_id = str(raw.get("id") or "").strip()
        if not domain or not item_id:
            raise ValueError("exemplar needs id and domain")
        loaded.append({
            "id": item_id,
            "domain": domain,
            "label": label,
            "features": _features_from_query({"features": raw.get("features")}),
            "source": str(raw.get("source") or "").strip(),
        })
        if not loaded[-1]["source"]:
            raise ValueError("exemplar needs source")
    return loaded


def assess_visual(
    query: dict[str, Any],
    *,
    memory: list[dict[str, Any]] | None = None,
    k: int = 5,
) -> SpecialistResult:
    domain = str(query.get("domain") or "").strip()
    if not domain:
        raise ValueError("query needs domain")
    features = _features_from_query(query)
    records = [
        item for item in (memory if memory is not None else load_exemplars())
        if item["domain"] == domain
    ]
    scored = sorted(
        ((_similarity(features, item["features"]), item) for item in records),
        key=lambda pair: pair[0],
        reverse=True,
    )
    nearest = [pair for pair in scored if pair[0] >= RETRIEVE_SIM][:k]
    neighbors = [
        {
            "id": item["id"],
            "label": item["label"],
            "similarity": round(sim, 4),
            "source": item["source"],
        }
        for sim, item in nearest
    ]
    if len(nearest) < MIN_NEIGHBORS:
        return SpecialistResult(
            SPECIALIST_ID,
            {
                "status": "abstain",
                "label": None,
                "reason": "unknown_visual_space",
                "domain": domain,
                "neighbors": neighbors,
                "features": {name: round(features[name], 4) for name in FEATURE_NAMES},
            },
            0.0,
            True,
            AuthorityMode.CONSULTATIVE,
            notes=(
                "Not enough known images in this domain.",
                "This is not the log anomaly_detector.",
            ),
        )
    mean_sim = sum(sim for sim, _ in nearest) / len(nearest)
    labels = {item["label"] for _, item in nearest}
    if len(labels) > 1:
        return SpecialistResult(
            SPECIALIST_ID,
            {
                "status": "abstain",
                "label": None,
                "reason": "mixed_visual_labels",
                "domain": domain,
                "neighbors": neighbors,
            },
            0.0,
            True,
            AuthorityMode.CONSULTATIVE,
            notes=("Mixed normal/novel neighbors do not create an incident.",),
        )
    if mean_sim < IN_SPACE_SIM:
        label = "novel"
        status = "suggest"
    else:
        label = nearest[0][1]["label"]
        status = "suggest"
    confidence = min(0.95, mean_sim if label == "normal" else max(0.55, 1.0 - mean_sim))
    return SpecialistResult(
        SPECIALIST_ID,
        {
            "status": status,
            "label": label,
            "domain": domain,
            "in_known_space": mean_sim >= IN_SPACE_SIM,
            "mean_similarity": round(mean_sim, 4),
            "neighbors": neighbors,
        },
        round(confidence, 4),
        False,
        AuthorityMode.CONSULTATIVE,
        notes=(
            "Consultative visual novelty only. No alarm is raised.",
            "Pixels are not stored.",
        ),
    )
