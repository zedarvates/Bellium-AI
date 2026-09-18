"""Animation timing report: easing, holds, peak and duplicate frames.

The measures are deterministic; the easing curve name comes from exemplar
retrieval (bellium/knn/easing-profile:v0) and must agree with the published
threshold rule. No frame is retimed, removed or exported.
"""

from __future__ import annotations

from typing import Any

from bellium.contracts.schema import AuthorityMode, SpecialistResult
from bellium.knn._motion import (
    DUPLICATE_MISMATCH,
    deterministic_easing_verdict,
    hold_runs,
    is_duplicate,
    motion_series,
    peak_index,
    progress_samples,
    spacing_variation,
)
from bellium.knn.easing_profile import FAMILIES, classify_easing

SPECIALIST_ID = "bellium/hybrid/animation-timing:v0"
MIN_TRANSITIONS = 3
ROUGH_SPACING = 0.6
NO_MOTION = 0.01


def _series(query: dict[str, Any]) -> list[float]:
    if query.get("series") is not None:
        return motion_series(query["series"], minimum=1)
    deltas = query.get("deltas")
    if not isinstance(deltas, (list, tuple)) or not deltas:
        raise ValueError("query needs a series or a list of deltas")
    values = []
    for index, delta in enumerate(deltas):
        if not isinstance(delta, dict) or delta.get("changed_ratio") is None:
            raise ValueError(f"delta {index} needs a changed_ratio; do not coerce to 0")
        values.append(delta["changed_ratio"])
    return motion_series(values, minimum=1)


def _mismatches(query: dict[str, Any]) -> list[float] | None:
    raw = query.get("frame_mismatches")
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)):
        raise ValueError("frame_mismatches must be a list")
    values = []
    for index, value in enumerate(raw):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"mismatch {index} must be numeric")
        values.append(float(value))
    return values


def report_timing(query: dict[str, Any]) -> SpecialistResult:
    family = str(query.get("family") or "").strip()
    if family not in FAMILIES:
        return SpecialistResult(
            SPECIALIST_ID,
            {"status": "abstain", "reason": "unknown_family", "family": family,
             "easing": None, "warnings": []},
            0.0, True, AuthorityMode.CONSULTATIVE,
            notes=("Unknown motion family; no timing report is invented.",),
        )
    series = _series(query)
    if len(series) < MIN_TRANSITIONS:
        return SpecialistResult(
            SPECIALIST_ID,
            {"status": "abstain", "reason": "clip_too_short", "family": family,
             "transitions": len(series), "easing": None, "warnings": []},
            0.0, True, AuthorityMode.CONSULTATIVE,
            notes=("Timing needs at least three transitions to mean anything.",),
        )
    holds = hold_runs(series)
    peak = peak_index(series)
    roughness = spacing_variation(series)
    # The easing describes the moving part: leading and trailing holds are
    # reported separately and would otherwise flatten the progress curve.
    moving = list(series)
    while moving and moving[0] <= 0.0:
        moving.pop(0)
    while moving and moving[-1] <= 0.0:
        moving.pop()
    samples = progress_samples(moving if len(moving) >= 3 else series)
    easing = classify_easing({"family": family, "samples": samples})
    baseline = deterministic_easing_verdict(samples)
    mismatches = _mismatches(query)
    duplicates: list[dict[str, Any]] = []
    if mismatches is not None:
        if len(mismatches) != len(series):
            raise ValueError("frame_mismatches must match the number of transitions")
        duplicates = [
            {"transition": index, "mismatch": round(value, 6)}
            for index, value in enumerate(mismatches)
            if is_duplicate(value)
        ]
    warnings: list[str] = []
    if max(series) < NO_MOTION:
        warnings.append("no_visible_motion")
    if duplicates:
        warnings.append("duplicate_frames")
    if roughness > ROUGH_SPACING:
        warnings.append("uneven_spacing")
    if peak == len(series) - 1 and len(series) > 2:
        warnings.append("peak_at_last_frame")
    if easing.abstained:
        warnings.append("easing_not_confirmed")
    else:
        trailing = holds[-1] if holds and holds[-1]["start"] + holds[-1]["length"] == len(series) else None
        if family in ("ui", "effect") and trailing is None:
            warnings.append("no_hold_at_end")
    confidence = 0.35 if warnings else 0.7
    if not easing.abstained:
        confidence = min(0.85, (easing.confidence or 0.0) + 0.2)
    return SpecialistResult(
        SPECIALIST_ID,
        {
            "status": "ready",
            "family": family,
            "transitions": len(series),
            "holds": holds,
            "peak": {"index": peak, "value": round(series[peak], 6)},
            "spacing_variation": round(roughness, 6),
            "progress_samples": [round(value, 6) for value in samples],
            "easing": easing.output.get("easing"),
            "easing_status": easing.output["status"],
            "easing_baseline": baseline,
            "easing_errors": easing.output.get("errors"),
            "duplicates": duplicates,
            "duplicate_limit": DUPLICATE_MISMATCH,
            "warnings": warnings,
            "certified": False,
            "frames_changed": 0,
        },
        round(confidence, 4),
        False,
        AuthorityMode.CONSULTATIVE,
        notes=(
            "Timing report only: nothing is retimed, trimmed or exported.",
            "Duplicate frames are reported with their mismatch, never removed.",
        ),
    )
