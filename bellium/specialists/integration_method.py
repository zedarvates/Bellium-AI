"""Consultative recommendation of an integrator for a normal field.

The recommendation never integrates and never returns a height: it names one of
the four published methods, reports the deterministic rule's own answer beside it,
and shows the measured errors of the exemplars that voted. A caller that disagrees
loses nothing but the time it chose to spend.
"""

from __future__ import annotations

from typing import Any

from bellium.contracts.schema import AuthorityMode, SpecialistResult
from bellium.knn.integration_method import (
    BASELINE_ID,
    SPECIALIST_ID,
    load_memory,
    recommend_method,
)

QUERY_KEYS = ("normals", "mask", "min_cosine", "neighbours", "memory")


def recommend_integration(query: dict[str, Any]) -> SpecialistResult:
    """Recommend one of least-squares, cumulative, cumulative-average or cumulative-vertical."""
    if not isinstance(query, dict):
        raise ValueError("query must be an object")
    unknown = sorted(set(query) - set(QUERY_KEYS))
    if unknown:
        raise ValueError(f"unknown query keys refuse to be ignored: {', '.join(unknown)}")
    normals = query.get("normals")
    if normals is None:
        raise ValueError("normals must be provided: there is no field to characterise")
    declared = query.get("memory")
    if declared is not None and not isinstance(declared, dict):
        raise ValueError("memory must be an object when provided")
    memory = declared if declared is not None else load_memory()
    options: dict[str, Any] = {}
    for key in ("min_cosine", "neighbours"):
        if key in query:
            options[key] = query[key]
    answer = recommend_method(normals, query.get("mask"), memory=memory, **options)
    shared: dict[str, Any] = {
        "method": answer["method"],
        "baseline_method": answer["baseline_method"],
        "baseline_id": BASELINE_ID,
        "agrees_with_baseline": answer["agrees_with_baseline"],
        "agreement": answer.get("agreement"),
        "features": answer["features"],
        "neighbours": answer["neighbours"],
        "exemplars": len(memory["items"]),
        "integrated": False,
        "certified": False,
        "pixels_changed": 0,
    }
    if answer["status"] == "abstain":
        return SpecialistResult(
            SPECIALIST_ID,
            {**shared, "status": "abstain", "reason": answer["reason"],
             "warnings": [answer["reason"]]},
            0.0, True, AuthorityMode.CONSULTATIVE,
            notes=(
                "Fewer than four pixels carry a slope, so the field cannot be characterised.",
                "No method was named and nothing was integrated.",
            ),
        )
    warnings: list[str] = []
    if answer["status"] == "baseline":
        warnings.append("not_enough_neighbours")
    if answer["agrees_with_baseline"] is False:
        warnings.append("learned_tier_overrides_the_published_rule")
    output = {
        **shared,
        "status": "ready",
        "reason": None,
        "recommended_by": "baseline" if answer["status"] == "baseline" else "knn",
        "warnings": warnings,
    }
    if answer["status"] == "baseline":
        return SpecialistResult(
            SPECIALIST_ID, output, 0.5, False, AuthorityMode.CONSULTATIVE,
            notes=(
                "No exemplar was similar enough, so the published rule answered instead.",
                "The features are returned so the caller can see why nothing matched.",
            ),
        )
    confidence = 0.5 + 0.3 * float(answer.get("agreement") or 0.0)
    return SpecialistResult(
        SPECIALIST_ID, output, round(confidence, 4), False, AuthorityMode.CONSULTATIVE,
        notes=(
            "A recommendation, not a measurement: no height was produced and no map was changed.",
            "Every neighbour carries the relative error of all four integrators on its fixture.",
        ),
    )

