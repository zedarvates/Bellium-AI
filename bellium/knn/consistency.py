"""Family-isolated visual consistency retrieval. Suggests a style id, never recolours."""

from __future__ import annotations

import json
from typing import Any

from bellium.contracts.schema import AuthorityMode, SpecialistResult
from bellium.knn.visual_anomaly import _features_from_query, _similarity
from bellium.resources import model_path

SPECIALIST_ID = "bellium/knn/consistency-retrieval:v0"
SCHEMA = "bellium.consistency-memory/v1"
FAMILIES = ("storycore-panel", "sprite", "texture")
MIN_NEIGHBORS = 3
MIN_SIMILARITY = 0.72
DEFAULT_MEMORY = model_path("knn", "visual", "consistency-v0.json")


def load_styles(path=None) -> list[dict[str, Any]]:
    payload = json.loads((path or DEFAULT_MEMORY).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise ValueError("consistency memory schema mismatch")
    items = payload.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("consistency memory needs items")
    loaded = []
    seen = set()
    for raw in items:
        if not isinstance(raw, dict):
            raise ValueError("style item must be an object")
        if raw.get("raw_image_stored") is True:
            raise ValueError("raw images must not be stored")
        item_id = str(raw.get("id") or "").strip()
        family = str(raw.get("family") or "").strip()
        style_id = str(raw.get("style_id") or "").strip()
        source = str(raw.get("source") or "").strip()
        if family not in FAMILIES:
            raise ValueError(f"family must be one of: {', '.join(FAMILIES)}")
        if not item_id or not style_id or not source:
            raise ValueError("item needs id, style_id and source")
        if item_id in seen:
            raise ValueError(f"duplicate style id: {item_id}")
        seen.add(item_id)
        loaded.append({
            "id": item_id,
            "family": family,
            "style_id": style_id,
            "source": source,
            "features": _features_from_query({"features": raw.get("features")}),
        })
    return loaded


def retrieve_consistency(
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
             "style_id": None, "neighbors": [], "recolor": False},
            0.0, True, AuthorityMode.CONSULTATIVE,
            notes=("Unknown family; no style is invented.",),
        )
    features = _features_from_query(query)
    records = [item for item in (memory if memory is not None else load_styles()) if item["family"] == family]
    scored = sorted(
        ((_similarity(features, item["features"]), item) for item in records),
        key=lambda pair: pair[0], reverse=True,
    )
    nearest = [pair for pair in scored if pair[0] >= MIN_SIMILARITY][:k]
    neighbors = [
        {"id": item["id"], "style_id": item["style_id"], "similarity": round(sim, 4),
         "source": item["source"]}
        for sim, item in nearest
    ]
    if len(nearest) < MIN_NEIGHBORS:
        return SpecialistResult(
            SPECIALIST_ID,
            {"status": "abstain", "reason": "too_few_similar_styles", "family": family,
             "style_id": None, "neighbors": neighbors, "recolor": False, "certified": False},
            0.0, True, AuthorityMode.CONSULTATIVE,
            notes=("Consistency retrieval abstains instead of inventing a look.",),
        )
    styles = {item["style_id"] for _, item in nearest}
    if len(styles) > 1:
        return SpecialistResult(
            SPECIALIST_ID,
            {"status": "abstain", "reason": "mixed_neighbor_styles", "family": family,
             "style_id": None, "neighbors": neighbors, "recolor": False, "certified": False},
            0.0, True, AuthorityMode.CONSULTATIVE,
            notes=("Disagreeing neighbors do not elect a visual style.",),
        )
    strength = min(0.9, sum(sim for sim, _ in nearest) / len(nearest))
    return SpecialistResult(
        SPECIALIST_ID,
        {
            "status": "suggest",
            "family": family,
            "style_id": next(iter(styles)),
            "neighbors": neighbors,
            "recolor": False,
            "certified": False,
        },
        round(strength, 4),
        False, AuthorityMode.CONSULTATIVE,
        notes=("A style id is a retrieval hint. Pixels are not changed.",),
    )

