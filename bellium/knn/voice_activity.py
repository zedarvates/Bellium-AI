"""k-NN voice activity on compact frame features. Audio is never stored."""

from __future__ import annotations

import json
import math
from typing import Any

from bellium.contracts.schema import AuthorityMode, SpecialistResult
from bellium.resources import model_path

SPECIALIST_ID = "bellium/knn/voice-activity:v0"
SCHEMA = "bellium.voice-activity-memory/v1"
FEATURE_NAMES = (
    "energy",
    "zcr",
    "periodicity",
    "crest",
    "high_ratio",
)
MIN_SAMPLES = 32
MIN_NEIGHBORS = 3
MIN_SIMILARITY = 0.70
DEFAULT_MEMORY = model_path("knn", "audio", "voice-activity-v0.json")


def extract_features(samples: object) -> dict[str, float]:
    if not isinstance(samples, list) or len(samples) < MIN_SAMPLES:
        raise ValueError(f"need at least {MIN_SAMPLES} samples")
    values: list[float] = []
    for item in samples:
        if item is None:
            raise ValueError("unknown samples must not be coerced to 0")
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError("samples must be numeric")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError("samples must be finite")
        values.append(max(-1.0, min(1.0, number)))
    count = len(values)
    rms = math.sqrt(sum(v * v for v in values) / count)
    energy = min(rms / 0.35, 1.0)
    crossings = sum(1 for i in range(1, count) if (values[i - 1] >= 0) != (values[i] >= 0))
    zcr = crossings / (count - 1)
    peak = max(abs(v) for v in values)
    crest = 0.0 if rms < 1e-8 else min(peak / (rms * 8.0), 1.0)
    diffs = [abs(values[i] - values[i - 1]) for i in range(1, count)]
    high_ratio = sum(1 for d in diffs if d > 0.08) / len(diffs)
    periodicity = 0.0
    if rms >= 1e-6:
        best = 0.0
        for lag in range(8, min(80, count // 2)):
            acc = sum(values[i] * values[i + lag] for i in range(count - lag)) / ((count - lag) * rms * rms)
            if acc > best:
                best = acc
        periodicity = min(max(best, 0.0), 1.0)
    return {
        "energy": energy,
        "zcr": min(zcr, 1.0),
        "periodicity": periodicity,
        "crest": crest,
        "high_ratio": min(high_ratio, 1.0),
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


def load_voice_frames(path=None) -> list[dict[str, Any]]:
    payload = json.loads((path or DEFAULT_MEMORY).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise ValueError("voice-activity memory schema mismatch")
    items = payload.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("voice-activity memory needs items")
    loaded = []
    for raw in items:
        if not isinstance(raw, dict):
            raise ValueError("item must be an object")
        if raw.get("raw_audio_stored") is True:
            raise ValueError("raw audio must not be stored")
        item_id = str(raw.get("id") or "").strip()
        label = str(raw.get("label") or "").strip()
        source = str(raw.get("source") or "").strip()
        if label not in {"speech", "silence"}:
            raise ValueError("label must be speech or silence")
        if not item_id or not source:
            raise ValueError("item needs id and source")
        loaded.append({
            "id": item_id,
            "label": label,
            "source": source,
            "features": _features(raw.get("features")),
        })
    return loaded


def classify_voice(
    query: dict[str, Any],
    *,
    memory: list[dict[str, Any]] | None = None,
    k: int = 5,
) -> SpecialistResult:
    if query.get("audio_path") is not None or query.get("waveform") is not None:
        raise ValueError("pass samples or features, not stored audio")
    if query.get("samples") is not None:
        features = extract_features(query["samples"])
    else:
        features = _features(query.get("features"))
    records = memory if memory is not None else load_voice_frames()
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
            {"status": "abstain", "reason": "too_few_similar_frames", "label": None,
             "neighbors": neighbors, "audio": None},
            0.0, True, AuthorityMode.CONSULTATIVE,
        )
    labels = {item["label"] for _, item in nearest}
    if len(labels) > 1:
        return SpecialistResult(
            SPECIALIST_ID,
            {"status": "abstain", "reason": "mixed_neighbor_labels", "label": None,
             "neighbors": neighbors, "audio": None},
            0.0, True, AuthorityMode.CONSULTATIVE,
            notes=("Disagreeing neighbors do not elect speech.",),
        )
    return SpecialistResult(
        SPECIALIST_ID,
        {"status": "suggest", "label": next(iter(labels)), "neighbors": neighbors,
         "audio": None, "certified": False},
        round(sum(sim for sim, _ in nearest) / len(nearest), 4),
        False, AuthorityMode.CONSULTATIVE,
        notes=("Frame classification only. Audio is not stored or transcribed.",),
    )

