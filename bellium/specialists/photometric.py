"""Consultative normals, albedo and relative height from controlled captures.

The light directions must be declared. Pixels that break the Lambertian model,
such as self-shadowed or specular ones, are reported through the fit residual and
the per-patch trust decision; nothing is ever written into the captures. The
height integrator consumes a normal field, its own output or any other, and
returns a relative surface: an additive constant is not recoverable from normals.
"""

from __future__ import annotations

import math
from typing import Any

from bellium.contracts.schema import AuthorityMode, SpecialistResult
from bellium.knn._image import shape
from bellium.material.controlled import from_image, to_gray
from bellium.material.integration import DEFAULT_METHOD, integrate_height
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

HEIGHT_SPECIALIST_ID = "bellium/hybrid/normal-to-height:v0"
HEIGHT_QUERY_KEYS = (
    "normals",
    "mask",
    "method",
    "iterations",
    "tolerance",
    "pixel_scale",
    "min_cosine",
)
FEW_VALID_RATIO = 0.5
# Measured on the controlled chain, captures then normals then height: refusing
# the pixels whose normal lies close to the image plane takes the relative error
# from 1.37 to 0.18 while keeping 96 % of them. The floor is a declared prior and
# a caller can pass min_cosine to change it or set it to zero.
GRAZING_FLOOR = 0.1


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


def recover_height(query: dict[str, Any]) -> SpecialistResult:
    """Relative height from a normal field, with the unrecoverable constant removed.

    The normal field may come from the sibling specialist or from anywhere else:
    this one only integrates, and it reports which integrator produced the numbers
    and whether that integrator finished.
    """
    if not isinstance(query, dict):
        raise ValueError("query must be an object")
    unknown = sorted(set(query) - set(HEIGHT_QUERY_KEYS))
    if unknown:
        raise ValueError(f"unknown query keys refuse to be ignored: {', '.join(unknown)}")
    normals = query.get("normals")
    if normals is None:
        raise ValueError("normals must be provided: a height is not inferable without slopes")
    options: dict[str, Any] = {"method": query.get("method", DEFAULT_METHOD)}
    for key in ("iterations", "tolerance", "pixel_scale"):
        if key in query:
            options[key] = query[key]
    options["min_cosine"] = query.get("min_cosine", GRAZING_FLOOR)
    result = integrate_height(normals, query.get("mask"), **options)
    rows = len(result["height"])
    columns = len(result["height"][0])
    total = rows * columns
    given_mask = query.get("mask")
    offered = 0
    for r in range(rows):
        for c in range(columns):
            if normals[r][c] is None:
                continue
            if given_mask is not None and given_mask[r][c] <= 0.0:
                continue
            offered += 1
    dropped = offered - result["pixels"]
    kept_ratio = result["pixels"] / offered if offered else 0.0
    valid_ratio = result["pixels"] / total if total else 0.0
    heights = [
        result["height"][r][c]
        for r in range(rows)
        for c in range(columns)
        if result["valid"][r][c]
    ]
    relief = round(max(heights) - min(heights), 6) if len(heights) > 1 else 0.0
    pixel_scale = result["pixel_scale"]
    shared: dict[str, Any] = {
        "method": result["method"],
        "pixel_scale": pixel_scale,
        "grazing_floor": result["min_cosine"],
        "offered": offered,
        "grazing_dropped": dropped,
        "height": result["height"],
        "valid": result["valid"],
        "pixels": result["pixels"],
        "valid_ratio": round(valid_ratio, 6),
        "relief": relief,
        "convergence": result["convergence"],
        "integrability": result["integrability"],
        "certified": False,
        "pixels_changed": 0,
        "note": result["note"],
    }
    if result["pixels"] == 0:
        return SpecialistResult(
            HEIGHT_SPECIALIST_ID,
            {
                **shared,
                "status": "abstain",
                "reason": "no_recoverable_slope",
                "offset_removed": 0.0,
                "warnings": ["no_recoverable_slope"],
            },
            0.0, True, AuthorityMode.CONSULTATIVE,
            notes=(
                "Every normal is missing or lies in the image plane, so no slope is recoverable.",
                "The field is returned as it stands; nothing was invented to fill it.",
            ),
        )
    warnings: list[str] = []
    if valid_ratio < FEW_VALID_RATIO:
        warnings.append("few_valid_pixels")
    if dropped > 0:
        warnings.append("grazing_pixels_dropped")
    convergence = result["convergence"]
    if convergence is not None and not convergence["converged"]:
        # The sweep count is a budget, not a promise: a smooth surface still has
        # large-scale error left when the sweeps stop.
        warnings.append("least_squares_not_converged")
    if relief == 0.0:
        warnings.append("flat_field")
    curl_verdict = result["integrability"]["verdict"]
    if curl_verdict in ("suspect", "not_integrable"):
        # No integrator can turn a rotational part into a surface; it is reported
        # rather than smoothed away.
        warnings.append("field_may_not_be_integrable")
    output = {
        **shared,
        "status": "ready",
        "reason": None,
        "units": "pixels" if pixel_scale == 1.0 else "declared-world-units",
        "offset_removed": result["offset_removed"],
        "warnings": warnings,
    }
    confidence = 0.8
    if "few_valid_pixels" in warnings or "least_squares_not_converged" in warnings:
        confidence = min(confidence, 0.4)
    if kept_ratio < FEW_VALID_RATIO:
        confidence = min(confidence, 0.4)
    if "flat_field" in warnings:
        confidence = min(confidence, 0.5)
    if curl_verdict == "suspect":
        confidence = min(confidence, 0.6)
    if curl_verdict == "not_integrable":
        confidence = min(confidence, 0.4)
    return SpecialistResult(
        HEIGHT_SPECIALIST_ID, output, round(confidence, 4), False, AuthorityMode.CONSULTATIVE,
        notes=(
            "A normal field fixes the height only up to an additive constant: that offset is "
            "removed and reported.",
            "The pixel-to-world scale is the declared pixel_scale, never inferred from the normals.",
            "Pixels below the grazing floor are refused and counted, not silently smoothed.",
            "The curl is reported because no integrator can turn it into a surface.",
        ),
    )
