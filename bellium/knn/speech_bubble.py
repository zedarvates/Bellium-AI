"""k-NN that labels a compact bright region as bubble or not. No OCR."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from bellium.contracts.schema import AuthorityMode, SpecialistResult
from bellium.resources import model_path

SPECIALIST_ID = "bellium/knn/speech-bubble-region:v0"
SCHEMA = "bellium.speech-bubble-memory/v1"
FEATURE_NAMES = (
    "fill",
    "aspect",
    "compactness",
    "brightness",
    "border_touch",
    "topness",
)
MIN_NEIGHBORS = 3
MIN_SIMILARITY = 0.80
DEFAULT_MEMORY = (
    model_path()
    / "knn"
    / "layout"
    / "speech-bubble-v0.json"
)


def _unit(name: str, value: object) -> float:
    if value is None:
        raise ValueError(f"feature {name} is unknown; do not coerce to 0")
    if isinstance(value, bool) and name == "border_touch":
        return 1.0 if value else 0.0
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"feature {name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"feature {name} must be between 0 and 1")
    return number


def bubble_features(source: dict[str, Any]) -> dict[str, float]:
    return {name: _unit(name, source[name] if name in source else None) for name in FEATURE_NAMES}


def _similarity(left: dict[str, float], right: dict[str, float]) -> float:
    distance = math.sqrt(sum((left[name] - right[name]) ** 2 for name in FEATURE_NAMES))
    return max(0.0, 1.0 - distance / math.sqrt(len(FEATURE_NAMES)))


def load_bubbles(path: str | Path | None = None) -> list[dict[str, Any]]:
    payload = json.loads(Path(path or DEFAULT_MEMORY).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise ValueError("speech bubble memory schema mismatch")
    items = payload.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("speech bubble memory needs items")
    loaded = []
    for raw in items:
        if not isinstance(raw, dict):
            raise ValueError("bubble item must be an object")
        if raw.get("raw_image_stored") is True or raw.get("text_stored") is True:
            raise ValueError("raw images and dialogue text must not be stored")
        item_id = str(raw.get("id") or "").strip()
        label = str(raw.get("label") or "").strip()
        source = str(raw.get("source") or "").strip()
        if label not in {"bubble", "not_bubble"}:
            raise ValueError("label must be bubble or not_bubble")
        if not item_id or not source:
            raise ValueError("item needs id and source")
        loaded.append({
            "id": item_id,
            "label": label,
            "source": source,
            "features": bubble_features(raw.get("features") or {}),
        })
    return loaded


def classify_region(
    query: dict[str, Any],
    *,
    memory: list[dict[str, Any]] | None = None,
    k: int = 5,
) -> SpecialistResult:
    features = bubble_features(query.get("features") or {})
    records = memory if memory is not None else load_bubbles()
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
            notes=("Disagreeing neighbors do not create a bubble.",),
        )
    label = next(iter(labels))
    return SpecialistResult(
        SPECIALIST_ID,
        {"status": "suggest", "label": label, "neighbors": neighbors, "text": None},
        round(sum(sim for sim, _ in nearest) / len(nearest), 4),
        False, AuthorityMode.CONSULTATIVE,
        notes=("No OCR. Dialogue is never read or stored.",),
    )
