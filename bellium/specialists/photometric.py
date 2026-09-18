"""Consultative normals and albedo from controlled multi-light captures.

The light directions must be declared. Pixels that break the Lambertian model,
such as self-shadowed or specular ones, are reported through the fit residual and
the per-patch trust decision; nothing is ever written into the captures.
"""

from __future__ import annotations

import math
from typing import Any

from bellium.contracts.schema import AuthorityMode, SpecialistResult
from bellium.knn._image import shape
from bellium.material.controlled import from_image, to_gray
from bellium.material.photometric import (
    photometric_normals,
    reliability_features,
    reliable_patches,
    validate_lights,
)

SPECIALIST_ID = "bellium/hybrid/photometric-normals:v0"
PATCH = 8
RESIDUAL_LIMIT = 0.02
MODEL_MISMATCH = 0.1
LOW_COVERAGE = 0.1


def _gamma(value: object) -> float:
    if value is None:
        return 1.0
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("gamma must be numeric")
    if not math.isfinite(float(value)) or float(value) <= 0.0:
        raise ValueError("gamma must be positive and finite")
    return float(value)


def _captures(raw: object, gamma: float) -> list[list[list[float]]]:
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError("captures must be a non-empty list of images")
    out = []
    for index, image in enumerate(raw):
        if not isinstance(image, list):
            raise ValueError(f"capture {index} must be an image")
        shape(image)
        gray = to_gray(from_image(image))
        if gamma == 1.0:
            out.append(gray)
        else:
            exponent = 1.0 / gamma
            out.append(
                [[max(value, 0.0) ** exponent for value in row] for row in gray]
            )
    return out


def recover_normals(query: dict[str, Any]) -> SpecialistResult:
    captures_raw = query.get("captures")
    lights_raw = query.get("lights")
    if lights_raw is None:
        raise ValueError("lights must be declared: the light directions are not inferable")
    lights = validate_lights(lights_raw)
    gamma = _gamma(query.get("gamma"))
    captures = _captures(captures_raw, gamma)
    if len(captures) != len(lights):
        raise ValueError("one capture per declared light direction is required")
    fit = photometric_normals(captures, lights)
    patches = reliable_patches(
        reliability_features(fit["residual"], captures, patch=PATCH),
        residual_limit=RESIDUAL_LIMIT,
    )
    if not patches:
        return SpecialistResult(
            SPECIALIST_ID,
            {"status": "abstain", "reason": "captures_too_small", "method": None,
             "normals": [], "albedo": [], "residual": [], "patches": [],
             "warnings": ["captures_too_small"], "certified": False,
             "pixels_changed": 0},
            0.0, True, AuthorityMode.CONSULTATIVE,
            notes=("The captures are too small for one patch; nothing is estimated.",),
        )
    trusted = sum(1 for patch in patches if patch["trusted"])
    valid = sum(
        1 for row in fit["normals"] for normal in row if normal is not None
    )
    total = len(fit["normals"]) * len(fit["normals"][0])
    residuals = [
        fit["residual"][r][c]
        for r in range(len(fit["residual"]))
        for c in range(len(fit["residual"][0]))
        if fit["normals"][r][c] is not None
    ]
    mean_residual = sum(residuals) / len(residuals) if residuals else 1.0
    warnings: list[str] = []
    if valid / total < 0.5:
        warnings.append("few_valid_pixels")
    coverage = trusted / len(patches)
    if coverage < LOW_COVERAGE:
        warnings.append("no_reliable_patch")
    if mean_residual > MODEL_MISMATCH:
        warnings.append("lambertian_mismatch")
    output = {
        "status": "ready",
        "method": fit["method"],
        "lights": len(lights),
        "gamma": gamma,
        "normals": [
            [None if normal is None else [round(value, 6) for value in normal] for normal in row]
            for row in fit["normals"]
        ],
        "albedo": [[round(value, 6) for value in row] for row in fit["albedo"]],
        "residual": [[round(value, 6) for value in row] for row in fit["residual"]],
        "patches": [
            {
                "row": patch["row"],
                "column": patch["column"],
                "trusted": patch["trusted"],
                "features": patch["features"],
            }
            for patch in patches
        ],
        "valid_ratio": round(valid / total, 6),
        "trusted_coverage": round(coverage, 6),
        "mean_residual": round(mean_residual, 6),
        "warnings": warnings,
        "certified": False,
        "pixels_changed": 0,
    }
    if "no_reliable_patch" in warnings or "lambertian_mismatch" in warnings:
        return SpecialistResult(
            SPECIALIST_ID, output, 0.0, True, AuthorityMode.CONSULTATIVE,
            notes=(
                "The captures do not follow the Lambertian model well enough to trust a normal.",
                "A constant specular term can bias the normals without raising the residual.",
            ),
        )
    confidence = max(0.2, min(0.85, 1.0 - output["mean_residual"] * 10.0))
    if "few_valid_pixels" in warnings:
        confidence = min(confidence, 0.4)
    return SpecialistResult(
        SPECIALIST_ID, output, round(confidence, 4), False, AuthorityMode.CONSULTATIVE,
        notes=(
            "Normals and albedo are estimates: the declared light directions are inputs, "
            "not measurements.",
            "Untrusted patches are reported, never silently smoothed.",
        ),
    )
