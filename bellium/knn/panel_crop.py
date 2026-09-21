"""Family-isolated k-NN for panel crop margins. Does not cut pixels itself."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from bellium.contracts.schema import AuthorityMode, SpecialistResult
from bellium.resources import model_path

SPECIALIST_ID = "bellium/knn/panel-safe-crop:v0"
SCHEMA = "bellium.panel-crop-memory/v1"
FAMILIES = ("storycore-panel", "sprite", "product")
FEATURE_NAMES = (
    "fill",
    "aspect",
    "touch_left",
    "touch_right",
    "touch_top",
    "touch_bottom",
)
MIN_NEIGHBORS = 3
MIN_SIMILARITY = 0.55
DEFAULT_MEMORY = (
    model_path()
    / "knn"
    / "layout"
    / "panel-crop-v0.json"
)


def _unit(name: str, value: object) -> float:
    if value is None:
        raise ValueError(f"feature {name} is unknown; do not coerce to 0")
    if isinstance(value, bool) and name.startswith("touch_"):
        return 1.0 if value else 0.0
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"feature {name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"feature {name} must be between 0 and 1")
    return number


def layout_features(source: dict[str, Any]) -> dict[str, float]:
    return {name: _unit(name, source[name] if name in source else None) for name in FEATURE_NAMES}


def _similarity(left: dict[str, float], right: dict[str, float]) -> float:
    distance = math.sqrt(sum((left[name] - right[name]) ** 2 for name in FEATURE_NAMES))
    return max(0.0, 1.0 - distance / math.sqrt(len(FEATURE_NAMES)))


def load_layouts(path: str | Path | None = None) -> list[dict[str, Any]]:
    payload = json.loads(Path(path or DEFAULT_MEMORY).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise ValueError("panel crop memory schema mismatch")
    items = payload.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("panel crop memory needs items")
    loaded = []
    for raw in items:
        if not isinstance(raw, dict):
            raise ValueError("layout item must be an object")
        if raw.get("raw_image_stored") is True:
            raise ValueError("raw images must not be stored")
        item_id = str(raw.get("id") or "").strip()
        family = str(raw.get("family") or "").strip()
        source = str(raw.get("source") or "").strip()
        verdict = str(raw.get("verdict") or "").strip()
        margin = raw.get("margin")
        if family not in FAMILIES:
            raise ValueError(f"family must be one of: {', '.join(FAMILIES)}")
        if verdict not in {"safe", "unsafe"}:
            raise ValueError("verdict must be safe or unsafe")
        if not item_id or not source:
            raise ValueError("layout item needs id and source")
        if isinstance(margin, bool) or not isinstance(margin, (int, float)):
            raise ValueError("margin must be numeric")
        number = float(margin)
        if not math.isfinite(number) or not 0.0 <= number <= 0.3:
            raise ValueError("margin must be between 0 and 0.3")
        loaded.append({
            "id": item_id,
            "family": family,
            "verdict": verdict,
            "margin": number,
            "source": source,
            "features": layout_features(raw.get("features") or {}),
        })
    return loaded


def propose_margin(
    query: dict[str, Any],
    *,
    memory: list[dict[str, Any]] | None = None,
    k: int = 5,
) -> SpecialistResult:
    family = str(query.get("family") or "").strip()
    if family not in FAMILIES:
        return SpecialistResult(
            SPECIALIST_ID,
            {"status": "abstain", "reason": "unknown_family", "family": family, "margin": None},
            0.0, True, AuthorityMode.CONSULTATIVE,
            notes=("Unknown layout family; no margin is invented.",),
        )
    features = layout_features(query.get("features") or {})
    records = [item for item in (memory if memory is not None else load_layouts()) if item["family"] == family]
    scored = sorted(
        ((_similarity(features, item["features"]), item) for item in records),
        key=lambda pair: pair[0], reverse=True,
    )
    nearest = [pair for pair in scored if pair[0] >= MIN_SIMILARITY][:k]
    neighbors = [
        {"id": item["id"], "verdict": item["verdict"], "margin": item["margin"],
         "similarity": round(sim, 4)}
        for sim, item in nearest
    ]
    if len(nearest) < MIN_NEIGHBORS:
        return SpecialistResult(
            SPECIALIST_ID,
            {"status": "abstain", "reason": "too_few_similar_layouts", "family": family,
             "margin": None, "neighbors": neighbors},
            0.0, True, AuthorityMode.CONSULTATIVE,
        )
    unsafe = sum(1 for _, item in nearest if item["verdict"] == "unsafe")
    if unsafe > len(nearest) / 2:
        return SpecialistResult(
            SPECIALIST_ID,
            {"status": "suggest", "family": family, "verdict": "unsafe",
             "margin": None, "neighbors": neighbors},
            round(sum(sim for sim, _ in nearest) / len(nearest), 4),
            False, AuthorityMode.CONSULTATIVE,
            notes=("Neighbors look like clipped layouts.",),
        )
    safe = [item for _, item in nearest if item["verdict"] == "safe"]
    if len(safe) < MIN_NEIGHBORS:
        return SpecialistResult(
            SPECIALIST_ID,
            {"status": "abstain", "reason": "too_few_safe_layouts", "family": family,
             "margin": None, "neighbors": neighbors},
            0.0, True, AuthorityMode.CONSULTATIVE,
        )
    margin = sum(item["margin"] for item in safe) / len(safe)
    return SpecialistResult(
        SPECIALIST_ID,
        {"status": "suggest", "family": family, "verdict": "safe",
         "margin": round(margin, 4), "neighbors": neighbors},
        round(sum(sim for sim, _ in nearest) / len(nearest), 4),
        False, AuthorityMode.CONSULTATIVE,
        notes=("k-NN proposes a margin only. It does not crop.",),
    )

