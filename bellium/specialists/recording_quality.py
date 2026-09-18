"""Clip-level recording quality. Discards samples after features."""

from __future__ import annotations

from typing import Any

from bellium.contracts.schema import AuthorityMode, SpecialistResult
from bellium.knn.recording_quality import classify_quality

SPECIALIST_ID = "bellium/hybrid/recording-quality:v0"


def assess_recording(query: dict[str, Any]) -> SpecialistResult:
    result = classify_quality(query)
    return SpecialistResult(
        SPECIALIST_ID,
        result.output,
        result.confidence,
        result.abstained,
        AuthorityMode.CONSULTATIVE,
        notes=result.notes + ("Consultative label only. Nothing is denoised.",),
    )

