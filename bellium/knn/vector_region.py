from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from bellium.contracts.schema import AuthorityMode, SpecialistResult
from bellium.resources import model_path

SPECIALIST_ID = "bellium/knn/vector-region:v0"
MEMORY_PATH = model_path("knn", "visual", "vector-regions-v0.json")
FEATURE_NAMES = ("luma_std", "unique_ratio", "compactness", "fill_ratio", "chroma")
MIN_SIMILARITY = 0.55


def load_vector_exemplars(path=None) -> list[dict[str, Any]]:
    target = Path(path) if path is not None else MEMORY_PATH
    if not target.is_file():
        return []
    data = json.loads(target.read_text(encoding="utf-8"))
    return list(data.get("regions", []))


def _unit(name: str, value: object) -> float:
    if value is None:
        raise ValueError(f"feature {name} is unknown; do not coerce to 0")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"feature {name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"feature {name} must be between 0 and 1")
    return number


def classify_vector_region(
    features: dict[str, Any],
    *,
    exemplars: list[dict[str, Any]] | None = None,
) -> SpecialistResult:
    vector = [_unit(name, features.get(name)) for name in FEATURE_NAMES]
    memory = exemplars if exemplars is not None else load_vector_exemplars()
    if not memory:
        return SpecialistResult(
            SPECIALIST_ID,
            {"status": "abstain", "reason": "no_exemplars"},
            0.0,
            True,
            AuthorityMode.CONSULTATIVE,
        )
    best_dist = float("inf")
    best = memory[0]
    for item in memory:
        feats = item["features"]
        dist = math.sqrt(sum((vector[i] - feats[name]) ** 2 for i, name in enumerate(FEATURE_NAMES)))
        if dist < best_dist:
            best_dist = dist
            best = item
    similarity = max(0.0, 1.0 - best_dist / math.sqrt(len(FEATURE_NAMES)))
    label = str(best.get("label") or "photo_texture")
    if similarity < MIN_SIMILARITY or label == "photo_texture":
        return SpecialistResult(
            SPECIALIST_ID,
            {
                "status": "abstain",
                "reason": "region_not_vector_safe",
                "label": label,
                "similarity": round(similarity, 4),
                "exemplar_id": best.get("id"),
            },
            round(similarity, 4),
            True,
            AuthorityMode.CONSULTATIVE,
            notes=("Textured or photographic regions stay raster; they are not traced as SVG.",),
        )
    return SpecialistResult(
        SPECIALIST_ID,
        {
            "status": "suggest",
            "label": label,
            "similarity": round(similarity, 4),
            "exemplar_id": best.get("id"),
        },
        round(min(0.96, similarity), 4),
        False,
        AuthorityMode.CONSULTATIVE,
        notes=("k-NN names a flat fill or a thin stroke; the contour tracer stays deterministic.",),
    )
