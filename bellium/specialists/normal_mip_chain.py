"""Level-of-detail chains for a normal field, with the drift they leave behind.

A chain is a reduction, and a reduction of unit vectors is not a colour resize: the
specialist builds the levels, reports what each one did to the field, and refuses to
return pixels it was not asked to materialize.
"""

from __future__ import annotations

from typing import Any

from bellium.contracts.schema import AuthorityMode, SpecialistResult
from bellium.material.resample import (
    DEFAULT_FACTOR,
    DEFAULT_LEVELS,
    DEFAULT_MAX_PIXELS,
    DEFAULT_METHOD,
    mip_chain,
)

SPECIALIST_ID = "bellium/hybrid/normal-mip-chain:v0"
QUERY_KEYS = ("normals", "mask", "levels", "factor", "method", "max_pixels")


def build_normal_mips(query: dict[str, Any]) -> SpecialistResult:
    """Reduce a normal field into levels and report the drift of the chain."""
    if not isinstance(query, dict):
        raise ValueError("query must be an object")
    unknown = sorted(set(query) - set(QUERY_KEYS))
    if unknown:
        raise ValueError(f"unknown query keys refuse to be ignored: {', '.join(unknown)}")
    normals = query.get("normals")
    if normals is None:
        raise ValueError("normals must be provided: there is no field to reduce")
    max_pixels = query.get("max_pixels", DEFAULT_MAX_PIXELS)
    if isinstance(max_pixels, bool) or not isinstance(max_pixels, int) or max_pixels < 1:
        raise ValueError("max_pixels must be a positive integer")
    chain = mip_chain(
        normals,
        mask=query.get("mask"),
        levels=query.get("levels", DEFAULT_LEVELS),
        factor=query.get("factor", DEFAULT_FACTOR),
        method=query.get("method", DEFAULT_METHOD),
    )
    warnings = list(chain["warnings"])
    too_large = chain["pixels_returned"] > max_pixels
    if too_large:
        # A chain of a large source is statistics, not a copy: returning hundreds of
        # megabytes of Python lists silently is worse than saying no.
        warnings.append("chain_too_large")
    shared: dict[str, Any] = {
        "method": chain["method"],
        "factor": chain["factor"],
        "levels": chain["levels"],
        "drift": chain["drift"],
        "flattening_limit": chain["flattening_limit"],
        "coverage_limit": chain["coverage_limit"],
        "pixels_returned": chain["pixels_returned"],
        "max_pixels": max_pixels,
        "pixels": None if too_large else chain["pixels"],
        "written_files": False,
        "certified": False,
        "pixels_changed": 0,
        "warnings": sorted(set(warnings)),
    }
    if too_large:
        return SpecialistResult(
            SPECIALIST_ID,
            {**shared, "status": "statistics_only", "reason": "chain_too_large"},
            0.5, False, AuthorityMode.CONSULTATIVE,
            notes=(
                "The level statistics are returned and the pixels are not, above the declared budget.",
                "Raise max_pixels deliberately rather than receiving the grids by accident.",
            ),
        )
    confidence = 0.8
    if "flattening_drift" in warnings:
        confidence = 0.5
    if "partial_coverage" in warnings:
        confidence = min(confidence, 0.6)
    return SpecialistResult(
        SPECIALIST_ID,
        {**shared, "status": "ready", "reason": None},
        round(confidence, 4), False, AuthorityMode.CONSULTATIVE,
        notes=(
            "Every level is a reduction of the level above it, and the drift is the change in mean Z.",
            "Nothing was written: the levels are returned as data for the caller.",
        ),
    )

