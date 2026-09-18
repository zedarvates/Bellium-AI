"""k-NN that names the easing curve of a motion series. No pixels are stored.

The memory holds the published reference curves of one family; retrieval answers
which curve the measured progress follows, with the per-curve error as evidence.
An ambiguous fit, or a fit that contradicts the published threshold rule,
abstains instead of guessing.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from bellium.contracts.schema import AuthorityMode, SpecialistResult
from bellium.knn._motion import (
    REFERENCE_CURVES,
    SAMPLES,
    curve_error,
    deterministic_easing_verdict,
    progress_samples,
)
from bellium.resources import model_path

SPECIALIST_ID = "bellium/knn/easing-profile:v0"
SCHEMA = "bellium.easing-profile-memory/v1"
FAMILIES = ("ui", "character", "effect")
LABELS = tuple(sorted(REFERENCE_CURVES))
FEATURE_NAMES = tuple(f"s{index}" for index in range(SAMPLES))
FIT_TOLERANCE = 0.06
FIT_MARGIN = 0.015
DEFAULT_MEMORY = model_path("knn", "motion", "easing-profiles-v0.json")


def _unit(name: str, value: object) -> float:
    if value is None:
        raise ValueError(f"sample {name} is unknown; do not coerce to 0")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"sample {name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0.0 or number > 1.5:
        raise ValueError(f"sample {name} must be between 0 and 1.5")
    return number


def _features(source: object) -> list[float]:
    if not isinstance(source, (list, tuple)) or len(source) != SAMPLES:
        raise ValueError(f"a progress curve needs exactly {SAMPLES} samples")
    return [_unit(name, value) for name, value in zip(FEATURE_NAMES, source)]


def load_profiles(path: str | Path | None = None) -> list[dict[str, Any]]:
    payload = json.loads(Path(path or DEFAULT_MEMORY).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise ValueError("easing profile memory schema mismatch")
    items = payload.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("easing profile memory needs items")
    loaded = []
    for raw in items:
        if not isinstance(raw, dict):
            raise ValueError("easing item must be an object")
        item_id = str(raw.get("id") or "").strip()
        family = str(raw.get("family") or "").strip()
        label = str(raw.get("label") or "").strip()
        source = str(raw.get("source") or "").strip()
        if family not in FAMILIES:
            raise ValueError(f"family must be one of: {', '.join(FAMILIES)}")
        if label not in LABELS:
            raise ValueError(f"label must be one of: {', '.join(LABELS)}")
        if not item_id or not source:
            raise ValueError("easing item needs id and source")
        loaded.append({
            "id": item_id,
            "family": family,
            "label": label,
            "source": source,
            "samples": _features(raw.get("samples")),
        })
    return loaded


def classify_easing(
    query: dict[str, Any],
    *,
    memory: list[dict[str, Any]] | None = None,
) -> SpecialistResult:
    family = str(query.get("family") or "").strip()
    if family not in FAMILIES:
        return SpecialistResult(
            SPECIALIST_ID,
            {"status": "abstain", "reason": "unknown_family", "family": family,
             "easing": None, "errors": {}},
            0.0, True, AuthorityMode.CONSULTATIVE,
            notes=("Unknown motion family; no curve is invented.",),
        )
    if query.get("series") is not None:
        samples = progress_samples(query["series"])
    else:
        samples = _features(query.get("samples"))
    baseline = deterministic_easing_verdict(samples)
    records = [item for item in (memory if memory is not None else load_profiles())
               if item["family"] == family]
    scored = sorted(
        ((curve_error(samples, item["samples"]), item) for item in records),
        key=lambda pair: pair[0],
    )
    errors = {item["label"]: round(error, 6) for error, item in scored}
    output: dict[str, Any] = {
        "family": family,
        "baseline": baseline,
        "samples": [round(value, 6) for value in samples],
        "errors": errors,
        "pixels_changed": 0,
    }
    if len(scored) < 2:
        output.update(status="abstain", reason="too_few_reference_curves", easing=None)
        return SpecialistResult(
            SPECIALIST_ID, output, 0.0, True, AuthorityMode.CONSULTATIVE,
        )
    best_error, best = scored[0]
    second_error = scored[1][0]
    output["fit_error"] = round(best_error, 6)
    output["margin"] = round(second_error - best_error, 6)
    if best_error > FIT_TOLERANCE:
        output.update(status="abstain", reason="no_curve_close_enough", easing=None)
        return SpecialistResult(
            SPECIALIST_ID, output, 0.0, True, AuthorityMode.CONSULTATIVE,
            notes=("The measured progress follows no published curve closely enough.",),
        )
    if second_error - best_error < FIT_MARGIN:
        output.update(status="abstain", reason="ambiguous_easing", easing=None)
        return SpecialistResult(
            SPECIALIST_ID, output, 0.0, True, AuthorityMode.CONSULTATIVE,
            notes=("Two reference curves fit equally well; no curve is elected.",),
        )
    if best["label"] != baseline:
        output.update(status="abstain", reason="baseline_disagrees", easing=None)
        return SpecialistResult(
            SPECIALIST_ID, output, 0.0, True, AuthorityMode.CONSULTATIVE,
            notes=("The nearest curve and the published threshold rule disagree.",),
        )
    confidence = max(0.0, min(0.9, 1.0 - best_error / FIT_TOLERANCE))
    output.update(status="suggest", easing=best["label"])
    return SpecialistResult(
        SPECIALIST_ID, output, round(confidence, 4), False, AuthorityMode.CONSULTATIVE,
        notes=(
            "Curve name from exemplar retrieval, with the per-curve error as evidence.",
            "Timing warnings and holds belong to the animation timing specialist.",
        ),
    )
