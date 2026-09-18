"""Language-isolated prosody k-NN. Syllable annotations only; no audio."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from bellium.contracts.schema import AuthorityMode, SpecialistResult
from bellium.resources import model_path
from bellium.language.evidence import parse_evidence, weakest_evidence

SPECIALIST_ID = "bellium/knn/prosody-profile:v0"
SCHEMA = "bellium.prosody-memory/v1"
FEATURE_NAMES = (
    "stress_ratio",
    "iamb_ratio",
    "trochee_ratio",
    "duration_cv",
    "final_lengthening",
    "late_stress",
)
LABELS = ("iambic", "trochaic", "even")
MIN_SYLLABLES = 2
MIN_NEIGHBORS = 3
MIN_SIMILARITY = 0.85
DEFAULT_MEMORY = (
    model_path()
    / "knn"
    / "language"
    / "prosody-v0.json"
)


def extract_features(syllables: object) -> dict[str, float]:
    if not isinstance(syllables, list) or len(syllables) < MIN_SYLLABLES:
        raise ValueError("need at least two annotated syllables")
    stresses: list[float] = []
    durations: list[float] = []
    for item in syllables:
        if not isinstance(item, dict):
            raise ValueError("each syllable must be an object")
        if "stress" not in item or "duration" not in item:
            raise ValueError("syllable needs stress and duration")
        if item["stress"] is None or item["duration"] is None:
            raise ValueError("unknown syllable fields must not be coerced to 0")
        if item["stress"] not in (True, False):
            raise ValueError("stress must be boolean")
        duration = item["duration"]
        if isinstance(duration, bool) or not isinstance(duration, (int, float)):
            raise ValueError("duration must be numeric")
        number = float(duration)
        if not math.isfinite(number) or not 0.0 < number <= 1.0:
            raise ValueError("duration must be between 0 exclusive and 1")
        stresses.append(1.0 if item["stress"] else 0.0)
        durations.append(number)
    count = len(stresses)
    pairs = count - 1
    iamb = sum(1 for i in range(pairs) if stresses[i] < stresses[i + 1]) / pairs
    trochee = sum(1 for i in range(pairs) if stresses[i] > stresses[i + 1]) / pairs
    mean_dur = sum(durations) / count
    variance = sum((value - mean_dur) ** 2 for value in durations) / count
    cv = min(math.sqrt(variance) / mean_dur, 1.0)
    stressed = [index for index, value in enumerate(stresses) if value]
    if stressed:
        late = (sum(stressed) / len(stressed)) / (count - 1)
    else:
        late = 0.5
    return {
        "stress_ratio": sum(stresses) / count,
        "iamb_ratio": iamb,
        "trochee_ratio": trochee,
        "duration_cv": cv,
        "final_lengthening": min(durations[-1] / (2.0 * mean_dur), 1.0),
        "late_stress": late,
    }


def _features_from_query(query: dict[str, Any]) -> dict[str, float]:
    if query.get("syllables") is not None:
        return extract_features(query["syllables"])
    source = query.get("features")
    if not isinstance(source, dict):
        raise ValueError("query needs syllables or features")
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
    distance = math.sqrt(sum((left[name] - right[name]) ** 2 for name in FEATURE_NAMES))
    return max(0.0, 1.0 - distance / math.sqrt(len(FEATURE_NAMES)))


def load_profiles(path: str | Path | None = None) -> list[dict[str, Any]]:
    payload = json.loads(Path(path or DEFAULT_MEMORY).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise ValueError("prosody memory schema mismatch")
    items = payload.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("prosody memory needs items")
    loaded = []
    for raw in items:
        if not isinstance(raw, dict):
            raise ValueError("prosody item must be an object")
        if raw.get("raw_audio_stored") is True:
            raise ValueError("raw audio must not be stored")
        item_id = str(raw.get("id") or "").strip()
        language_id = str(raw.get("language_id") or "").strip()
        source = str(raw.get("source") or "").strip()
        label = str(raw.get("label") or "").strip()
        if not item_id or not language_id or not source:
            raise ValueError("prosody item needs id, language_id and source")
        if label not in LABELS:
            raise ValueError("label must be iambic, trochaic or even")
        loaded.append({
            "id": item_id,
            "language_id": language_id,
            "label": label,
            "evidence": parse_evidence(raw.get("evidence")),
            "source": source,
            "features": _features_from_query({"features": raw.get("features")}),
        })
    return loaded


def retrieve_prosody(
    query: dict[str, Any],
    *,
    memory: list[dict[str, Any]] | None = None,
    k: int = 5,
) -> SpecialistResult:
    language_id = str(query.get("language_id") or "").strip()
    if not language_id:
        raise ValueError("query needs language_id")
    features = _features_from_query(query)
    records = [
        item for item in (memory if memory is not None else load_profiles())
        if item["language_id"] == language_id
    ]
    scored = sorted(
        ((_similarity(features, item["features"]), item) for item in records),
        key=lambda pair: pair[0],
        reverse=True,
    )
    nearest = [pair for pair in scored if pair[0] >= MIN_SIMILARITY][:k]
    neighbors = [
        {
            "id": item["id"],
            "label": item["label"],
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
                "reason": "too_few_similar_profiles",
                "language_id": language_id,
                "label": None,
                "certified": False,
                "evidence": "speculative",
                "neighbors": neighbors,
            },
            0.0,
            True,
            AuthorityMode.CONSULTATIVE,
            notes=("Prosody k-NN abstains instead of inventing a language rhythm.",),
        )
    labels = {item["label"] for _, item in nearest}
    if len(labels) > 1:
        return SpecialistResult(
            SPECIALIST_ID,
            {
                "status": "abstain",
                "reason": "mixed_neighbor_profiles",
                "language_id": language_id,
                "label": None,
                "certified": False,
                "evidence": "speculative",
                "neighbors": neighbors,
            },
            0.0,
            True,
            AuthorityMode.CONSULTATIVE,
            notes=("Disagreeing rhythm neighbors do not elect a profile.",),
        )
    evidence = weakest_evidence([item["evidence"] for _, item in nearest])
    strength = min(0.9, sum(sim for sim, _ in nearest) / len(nearest))
    return SpecialistResult(
        SPECIALIST_ID,
        {
            "status": "suggest",
            "language_id": language_id,
            "label": next(iter(labels)),
            "evidence": evidence,
            "certified": False,
            "neighbors": neighbors,
        },
        round(strength, 4),
        False,
        AuthorityMode.CONSULTATIVE,
        notes=(
            "A matching contour is not a speaker recording.",
            "certified stays false: this is a profile suggestion.",
        ),
    )
