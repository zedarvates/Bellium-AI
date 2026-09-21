"""k-NN precedent retrieval for the consequence of a proposed action.

Family-isolated memories of verified precedents. A precedent is a report, not
an authorization: nothing here executes anything.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from bellium.contracts.schema import AuthorityMode, SpecialistResult
from bellium.resources import model_path

SPECIALIST_ID = "bellium/knn/consequence-precedent:v0"
BASELINE_ID = "bellium/deterministic/consequence-risk:v0"
SCHEMA = "bellium.consequence-precedent-memory/v1"
FAMILIES = ("filesystem", "asset-pipeline", "engine", "deployment", "network")
LABELS = ("safe", "review", "dangerous")
FEATURE_NAMES = (
    "reversible",
    "scope_ratio",
    "touches_protected",
    "has_backup",
    "dry_run_available",
    "blast_radius",
    "idempotent",
    "recent_failures",
)
MIN_NEIGHBORS = 3
MIN_SIMILARITY = 0.72
DEFAULT_MEMORY = model_path("knn", "tools", "consequence-precedents-v0.json")
FLAG_FEATURES = {"reversible", "touches_protected", "has_backup", "dry_run_available", "idempotent"}


def _unit(name: str, value: object) -> float:
    if value is None:
        raise ValueError(f"feature {name} is unknown; do not coerce to 0")
    if isinstance(value, bool):
        if name in FLAG_FEATURES:
            return 1.0 if value else 0.0
        raise ValueError(f"feature {name} must be numeric, not boolean")
    if not isinstance(value, (int, float)):
        raise ValueError(f"feature {name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"feature {name} must be between 0 and 1")
    return number


def consequence_features(source: object) -> dict[str, float]:
    if not isinstance(source, dict):
        raise ValueError("features must be an object")
    extra = set(source) - set(FEATURE_NAMES)
    if extra:
        raise ValueError(f"unknown features: {', '.join(sorted(extra))}")
    return {name: _unit(name, source[name] if name in source else None) for name in FEATURE_NAMES}


def risk_score(features: dict[str, float]) -> float:
    """Weighted risk score in [0, 1]. Published and cheap to recompute."""
    return (
        0.28 * (1.0 - features["reversible"])
        + 0.20 * features["blast_radius"]
        + 0.18 * (1.0 - features["has_backup"])
        + 0.14 * features["scope_ratio"]
        + 0.10 * features["touches_protected"]
        + 0.05 * (1.0 - features["idempotent"])
        + 0.05 * features["recent_failures"]
    )


def deterministic_risk_label(features: dict[str, float], *, review_limit: float = 0.25,
                             dangerous_limit: float = 0.45) -> str:
    """Published baseline kept beside the precedents and the micro-NN."""
    if features["touches_protected"] >= 0.5 and features["has_backup"] < 0.5:
        return "dangerous"
    score = risk_score(features)
    if score >= dangerous_limit:
        return "dangerous"
    if score >= review_limit:
        return "review"
    return "safe"


def _similarity(left: dict[str, float], right: dict[str, float]) -> float:
    distance = math.sqrt(sum((left[name] - right[name]) ** 2 for name in FEATURE_NAMES))
    return max(0.0, 1.0 - distance / math.sqrt(len(FEATURE_NAMES)))


def load_precedents(path: str | Path | None = None) -> list[dict[str, Any]]:
    payload = json.loads(Path(path or DEFAULT_MEMORY).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise ValueError("consequence precedent memory schema mismatch")
    items = payload.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("consequence precedent memory needs items")
    return _validate_precedents(items)


def _validate_precedents(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The disk and injected-memory paths share the same evidence contract."""
    if not isinstance(items, list):
        raise ValueError("precedents must be a list")
    loaded = []
    seen = set()
    for raw in items:
        if not isinstance(raw, dict):
            raise ValueError("precedent must be an object")
        item_id = str(raw.get("id") or "").strip()
        family = str(raw.get("family") or "").strip()
        label = str(raw.get("label") or "").strip()
        source = str(raw.get("source") or "").strip()
        if family not in FAMILIES:
            raise ValueError(f"family must be one of: {', '.join(FAMILIES)}")
        if label not in LABELS:
            raise ValueError(f"label must be one of: {', '.join(LABELS)}")
        if not item_id or not source:
            raise ValueError("precedent needs id and source")
        if item_id in seen:
            raise ValueError(f"duplicate precedent id: {item_id}")
        seen.add(item_id)
        verified = raw.get("verified", False)
        if not isinstance(verified, bool):
            raise ValueError("verified must be a boolean")
        loaded.append({
            "id": item_id,
            "family": family,
            "label": label,
            "source": source,
            "verified": verified,
            "features": consequence_features(raw.get("features")),
        })
    return loaded


def retrieve_precedent(
    query: dict[str, Any],
    *,
    memory: list[dict[str, Any]] | None = None,
    k: int = 5,
) -> SpecialistResult:
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    family = str(query.get("family") or "").strip()
    if family not in FAMILIES:
        return SpecialistResult(
            SPECIALIST_ID,
            {"status": "abstain", "reason": "unknown_family", "family": family,
             "label": None, "neighbors": []},
            0.0, True, AuthorityMode.CONSULTATIVE,
            notes=("Unknown action family; no consequence is invented.",),
        )
    features = consequence_features(query.get("features"))
    records = [item for item in (_validate_precedents(memory) if memory is not None else load_precedents())
               if item["family"] == family]
    scored = sorted(
        ((_similarity(features, item["features"]), item) for item in records),
        key=lambda pair: pair[0], reverse=True,
    )
    nearest = [pair for pair in scored if pair[0] >= MIN_SIMILARITY][:k]
    verified = [pair for pair in nearest if pair[1]["verified"]]
    neighbors = [
        {"id": item["id"], "label": item["label"], "verified": item["verified"],
         "similarity": round(sim, 4), "source": item["source"]}
        for sim, item in nearest
    ]
    output = {"family": family, "neighbors": neighbors, "executes": False}
    if len(verified) < MIN_NEIGHBORS:
        output.update(status="abstain", reason="too_few_verified_precedents", label=None)
        return SpecialistResult(
            SPECIALIST_ID, output, 0.0, True, AuthorityMode.CONSULTATIVE,
            notes=("Unverified or missing precedents do not decide anything.",),
        )
    labels = {item["label"] for _, item in verified}
    if len(labels) > 1:
        output.update(status="abstain", reason="mixed_precedent_labels", label=None)
        return SpecialistResult(
            SPECIALIST_ID, output, 0.0, True, AuthorityMode.CONSULTATIVE,
            notes=("Disagreeing precedents do not elect a consequence.",),
        )
    strength = sum(sim for sim, _ in verified) / len(verified)
    output.update(status="suggest", label=next(iter(labels)))
    return SpecialistResult(
        SPECIALIST_ID, output, round(min(0.9, strength), 4), False, AuthorityMode.CONSULTATIVE,
        notes=("A precedent report only. Deterministic policy still owns execution.",),
    )
