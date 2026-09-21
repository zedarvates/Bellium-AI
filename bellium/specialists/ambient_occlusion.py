"""Consultative ambient occlusion estimation from height or normal fields.

Produces an accessibility map in [0.0, 1.0] by ray-marching horizon elevation angles
across discrete radial slices. No assets or files are touched or modified.
"""

from __future__ import annotations

from typing import Any

from bellium.contracts.schema import AuthorityMode, SpecialistResult
from bellium.material.ambient_occlusion import (
    DEFAULT_DIRECTIONS,
    DEFAULT_RADIUS,
    horizon_ambient_occlusion,
)
from bellium.material.integration import integrate_height

SPECIALIST_ID = "bellium/hybrid/ambient-occlusion:v0"
QUERY_KEYS = (
    "height",
    "normals",
    "mask",
    "radius",
    "directions",
    "pixel_scale",
    "elevation_scale",
)


def estimate_ambient_occlusion(query: dict[str, Any]) -> SpecialistResult:
    """Estimate ambient occlusion factor from height map or normal field."""
    if not isinstance(query, dict):
        raise ValueError("query must be an object")
    unknown = sorted(set(query) - set(QUERY_KEYS))
    if unknown:
        raise ValueError(f"unknown query keys refuse to be ignored: {', '.join(unknown)}")

    height = query.get("height")
    normals = query.get("normals")
    mask = query.get("mask")

    if height is None and normals is None:
        raise ValueError("either 'height' or 'normals' must be provided")

    pixel_scale = query.get("pixel_scale", 1.0)
    elevation_scale = query.get("elevation_scale", 1.0)
    radius = query.get("radius", DEFAULT_RADIUS)
    directions = query.get("directions", DEFAULT_DIRECTIONS)

    source_kind = "height"
    if height is None:
        # Integrated from normals
        source_kind = "integrated_from_normals"
        integrated = integrate_height(normals, mask, pixel_scale=pixel_scale)
        height = integrated["height"]
        mask = integrated["valid"]

    result = horizon_ambient_occlusion(
        height,
        mask,
        radius=radius,
        directions=directions,
        pixel_scale=pixel_scale,
        elevation_scale=elevation_scale,
    )

    warnings: list[str] = []
    if result["pixels"] == 0:
        return SpecialistResult(
            SPECIALIST_ID,
            {
                "status": "abstain",
                "reason": "no_valid_pixels",
                "ao": result["ao"],
                "valid": result["valid"],
                "mean_ao": 0.0,
                "pixels": 0,
                "radius": radius,
                "directions": directions,
                "source_kind": source_kind,
                "written_files": False,
                "certified": False,
                "pixels_changed": 0,
                "warnings": ["no_valid_pixels"],
            },
            0.0,
            True,
            AuthorityMode.CONSULTATIVE,
            notes=("No valid pixel present in height or normal map.",),
        )

    if result["mean_ao"] > 0.99:
        warnings.append("nearly_flat_field")

    shared: dict[str, Any] = {
        "status": "ready",
        "reason": None,
        "ao": result["ao"],
        "valid": result["valid"],
        "mean_ao": result["mean_ao"],
        "pixels": result["pixels"],
        "radius": radius,
        "directions": directions,
        "pixel_scale": float(pixel_scale),
        "elevation_scale": float(elevation_scale),
        "source_kind": source_kind,
        "written_files": False,
        "certified": False,
        "pixels_changed": 0,
        "warnings": warnings,
    }

    confidence = 0.85
    if source_kind == "integrated_from_normals":
        confidence = 0.75
    if "nearly_flat_field" in warnings:
        confidence = min(confidence, 0.6)

    return SpecialistResult(
        SPECIALIST_ID,
        shared,
        round(confidence, 4),
        False,
        AuthorityMode.CONSULTATIVE,
        notes=(
            "Ambient occlusion represents horizon accessibility in [0.0, 1.0].",
            "The calculation is purely geometric along discrete radial horizon rays.",
            "Scale ratio between horizontal distance and vertical elevation is declared by caller.",
        ),
    )

