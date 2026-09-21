"""Consultative albedo and illumination separation for controlled or declared inputs.

The image alone cannot say whether its low-frequency variation is shading or
albedo. The caller declares its prior; without one the report exposes both
candidates and the evidence instead of claiming a separation.
"""

from __future__ import annotations

import math
from typing import Any

from bellium.contracts.schema import AuthorityMode, SpecialistResult
from bellium.knn._image import Image, shape
from bellium.material.controlled import from_image
from bellium.material.separation import (
    CLIPPED_ABSTENTION,
    DEFAULT_RADIUS,
    SHADING_PRIORS,
    separate_render,
)

SPECIALIST_ID = "bellium/hybrid/albedo-separation:v0"
FLAT_DETAIL_LIMIT = 0.05


def _gamma(value: object) -> float:
    if value is None:
        return 1.0
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("gamma must be numeric")
    if not math.isfinite(float(value)) or float(value) <= 0.0:
        raise ValueError("gamma must be positive and finite")
    return float(value)


def linearize(image: Image, gamma: float) -> list[list[tuple[float, float, float]]]:
    """Undo a display gamma before separating, so the product model holds."""
    linear = from_image(image)
    if gamma == 1.0:
        return linear
    exponent = 1.0 / gamma
    return [
        [
            (
                pixel[0] ** exponent,
                pixel[1] ** exponent,
                pixel[2] ** exponent,
            )
            for pixel in row
        ]
        for row in linear
    ]


def separate_image(query: dict[str, Any]) -> SpecialistResult:
    image = query.get("image")
    if image is None:
        raise ValueError("query needs image")
    shape(image)
    prior = str(query.get("expected_shading") or "unknown").strip()
    if prior not in SHADING_PRIORS:
        raise ValueError(f"expected_shading must be one of: {', '.join(SHADING_PRIORS)}")
    radius = query.get("radius", DEFAULT_RADIUS)
    if isinstance(radius, bool) or not isinstance(radius, int) or radius < 1:
        raise ValueError("radius must be a positive integer")
    gamma = _gamma(query.get("gamma"))
    report = separate_render(
        linearize(image, gamma), expected_shading=prior, radius=radius
    )
    warnings: list[str] = []
    if report["evidence"]["clipped_ratio"] >= CLIPPED_ABSTENTION:
        warnings.append("highlights_clipped")
    if report["evidence"]["detail_energy"] < FLAT_DETAIL_LIMIT:
        warnings.append("flat_albedo_not_identifiable")
    if report["unresolved"]:
        warnings.append("shading_prior_required")
    output = {
        "status": "abstain" if "highlights_clipped" in warnings else "ready",
        "albedo": report["albedo"],
        "illumination": report["illumination"],
        "method": report["method"],
        "choice": report["choice"],
        "recommendation": report["recommendation"],
        "unresolved": report["unresolved"],
        "evidence": report["evidence"],
        "radius": radius,
        "gamma": gamma,
        "channel_anchors": report["channel_anchors"],
        "warnings": warnings,
        "certified": False,
        "pixels_changed": 0,
        "candidates": report["candidates"],
    }
    if "highlights_clipped" in warnings:
        return SpecialistResult(
            SPECIALIST_ID, output, 0.0, True, AuthorityMode.CONSULTATIVE,
            notes=(
                "Clipped highlights destroy the product model; the image is not separated.",
                "The illumination level is never identifiable from a single image anyway.",
            ),
        )
    confidence = 0.75 if not report["unresolved"] else 0.4
    return SpecialistResult(
        SPECIALIST_ID, output, confidence, False, AuthorityMode.CONSULTATIVE,
        notes=(
            "Albedo and illumination are estimates, not measurements: both candidates "
            "are returned with the evidence that separates them.",
            "A declared shading prior determines the answer; without one the choice stays open.",
        ),
    )
