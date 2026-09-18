"""Language-isolated phoneme k-NN. Missing formants stay missing."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from bellium.contracts.schema import AuthorityMode, SpecialistResult
from bellium.resources import model_path
from bellium.language.evidence import parse_evidence, weakest_evidence

SPECIALIST_ID = "bellium/knn/language-phoneme:v0"
SCHEMA = "bellium.phoneme-inventory/v1"
FEATURE_ORDER = (
    "voicing",
    "manner",
    "place",
    "height",
    "backness",
    "rounding",
    "f1",
    "f2",
)
MIN_NEIGHBORS = 3
MIN_SIMILARITY = 0.50
DEFAULT_INVENTORY = (
    model_path()
    / "knn"
    / "language"
    / "phoneme-inventory-v0.json"
)


def _finite_unit(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"feature {name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"feature {name} must be between 0 and 1")
    return number


def _features(source: object) -> dict[str, float]:
    if not isinstance(source, dict) or not source:
        raise ValueError("features must be a non-empty object")
    extra = [key for key in source if key not in FEATURE_ORDER]
    if extra:
        raise ValueError(f"unknown phoneme features: {', '.join(extra)}")
    values: dict[str, float] = {}
    for name, raw in source.items():
        if raw is None:
            continue
        values[name] = _finite_unit(name, raw)
    if not values:
        raise ValueError("at least one phoneme feature is required")
    return values


def _item(raw: object) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("inventory item must be an object")
    item_id = str(raw.get("id") or "").strip()
    ipa = str(raw.get("ipa") or "").strip()
    language_id = str(raw.get("language_id") or "").strip()
    source = str(raw.get("source") or "").strip()
    if not item_id or not ipa or not language_id or not source:
        raise ValueError("item needs id, ipa, language_id and source")
    return {
        "id": item_id,
        "ipa": ipa,
        "language_id": language_id,
        "evidence": parse_evidence(raw.get("evidence")),
        "source": source,
        "features": _features(raw.get("features")),
    }


def load_inventory(path: str | Path | None = None) -> list[dict[str, Any]]:
    payload = json.loads(Path(path or DEFAULT_INVENTORY).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise ValueError("phoneme inventory schema mismatch")
    language_id = str(payload.get("language_id") or "").strip()
    items = payload.get("items")
    if not language_id or not isinstance(items, list) or not items:
        raise ValueError("inventory needs language_id and items")
    loaded = []
    seen = set()
    for raw in items:
        item = _item(raw)
        if item["language_id"] != language_id:
            raise ValueError("item language differs from inventory language")
        if item["id"] in seen:
            raise ValueError("duplicate phoneme id")
        seen.add(item["id"])
        loaded.append(item)
    return loaded


def _similarity(query: dict[str, float], other: dict[str, float]) -> float | None:
    shared = [name for name in FEATURE_ORDER if name in query and name in other]
    if len(shared) < 3:
        return None
    distance = math.sqrt(sum((query[name] - other[name]) ** 2 for name in shared))
    return max(0.0, 1.0 - distance / math.sqrt(len(shared)))


def retrieve_phoneme(
    query: dict[str, Any],
    *,
    inventory: list[dict[str, Any]] | None = None,
    k: int = 5,
    min_similarity: float = MIN_SIMILARITY,
) -> SpecialistResult:
    language_id = str(query.get("language_id") or "").strip()
    if not language_id:
        raise ValueError("query needs language_id")
    features = _features(query.get("features"))
    records = inventory if inventory is not None else load_inventory()
    family = [item for item in records if item["language_id"] == language_id]
    scored: list[tuple[float, dict[str, Any]]] = []
    for item in family:
        sim = _similarity(features, item["features"])
        if sim is None or sim < float(min_similarity):
            continue
        scored.append((sim, item))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    nearest = scored[:k]
    neighbors = [
        {
            "id": item["id"],
            "ipa": item["ipa"],
            "similarity": round(sim, 4),
            "evidence": item["evidence"],
            "source": item["source"],
        }
        for sim, item in nearest
    ]
    if len(nearest) < MIN_NEIGHBORS:
        return SpecialistResult(
            SPECIALIST_ID,
            {
                "status": "abstain",
                "language_id": language_id,
                "evidence": "speculative",
                "certified": False,
                "neighbors": neighbors,
                "reason": "too_few_similar_attested_or_labelled_phones",
            },
            0.0,
            True,
            AuthorityMode.CONSULTATIVE,
            notes=("Phoneme k-NN abstains instead of inventing a phone.",),
        )
    evidence = weakest_evidence([item["evidence"] for _, item in nearest])
    best_similarity = nearest[0][0]
    best_phones = {item["ipa"] for sim, item in scored if math.isclose(sim, best_similarity, abs_tol=1e-12)}
    if len(best_phones) > 1:
        return SpecialistResult(
            SPECIALIST_ID,
            {"status": "abstain", "reason": "ambiguous_nearest_phones", "language_id": language_id,
             "ipa": None, "certified": False, "evidence": "inferred", "neighbors": neighbors},
            0.0, True, AuthorityMode.CONSULTATIVE,
        )
    certified = evidence == "attested" and features == nearest[0][1]["features"]
    if not certified and evidence == "attested":
        evidence = "inferred"
    proposal = nearest[0][1]["ipa"]
    strength = min(0.95, sum(sim for sim, _ in nearest) / len(nearest))
    return SpecialistResult(
        SPECIALIST_ID,
        {
            "status": "suggest",
            "language_id": language_id,
            "ipa": proposal,
            "evidence": evidence,
            "certified": certified,
            "neighbors": neighbors,
        },
        round(strength, 4),
        False,
        AuthorityMode.CONSULTATIVE,
        notes=(
            "Proposal is retrieval, not a reconstructed language.",
            "Certification requires a unique exact feature match and attested support.",
        ),
    )
