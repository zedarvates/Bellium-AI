"""Normal-map handedness: inspect what a map states, convert what a caller declares.

The specialist reports the symptoms a map states about itself, including a
measurement of which handedness makes the field integrable, and it converts only
what the caller declares: a detection is evidence, not a licence to rewrite an
asset. It abstains when the input is not a normal map at all. Nothing is written
to disk.
"""

from __future__ import annotations

from typing import Any

from bellium.contracts.schema import AuthorityMode, SpecialistResult
from bellium.material.normals_io import (
    CONVENTIONS,
    flip_normal_convention,
    inspect_normal_map,
)

SPECIALIST_ID = "bellium/hybrid/normal-map-convention:v0"
QUERY_KEYS = ("image", "mask", "from_convention", "to_convention")
CONFIDENCE_BY_VERDICT = {
    "tangent-space": 0.8,
    "ambiguous": 0.5,
    "inverted-z": 0.6,
    "object-space": 0.6,
}


def convert_normal_map(query: dict[str, Any]) -> SpecialistResult:
    if not isinstance(query, dict):
        raise ValueError("query must be an object")
    unknown = sorted(set(query) - set(QUERY_KEYS))
    if unknown:
        raise ValueError(f"unknown query keys refuse to be ignored: {', '.join(unknown)}")
    image = query.get("image")
    if image is None:
        raise ValueError("image must be provided: there is nothing to inspect")
    declaration = query.get("from_convention", "+y")
    if declaration not in CONVENTIONS:
        raise ValueError(f"from_convention must be one of: {', '.join(CONVENTIONS)}")
    target = query.get("to_convention")
    if target is not None and target not in CONVENTIONS:
        raise ValueError(f"to_convention must be one of: {', '.join(CONVENTIONS)}")
    mask = query.get("mask")
    report = inspect_normal_map(image, convention=declaration, mask=mask)
    if report["status"] == "abstain":
        return SpecialistResult(
            SPECIALIST_ID,
            {
                "status": "abstain",
                "reason": report["reason"],
                "verdict": None,
                "converted": None,
                "y_convention_detection": report["y_convention_detection"],
                "written_files": False,
                "certified": False,
                "pixels_changed": 0,
                "warnings": [report["reason"]],
            },
            0.0, True, AuthorityMode.CONSULTATIVE,
            notes=("Every pixel decoded to a zero vector, so there is nothing to convert.",),
        )
    shared: dict[str, Any] = {
        "verdict": report["verdict"],
        "from_convention": declaration,
        "to_convention": target,
        "mean_z": report["mean_z"],
        "negative_z_ratio": report["negative_z_ratio"],
        "mean_non_unit": report["mean_non_unit"],
        "saturation_ratio": report["saturation_ratio"],
        "pixels": report["pixels"],
        "y_convention_decidable": report["y_convention_decidable"],
        "y_convention_detection": report["y_convention_detection"],
        "written_files": False,
        "certified": False,
        "pixels_changed": 0,
    }
    if report["verdict"] == "not-a-normal-map":
        return SpecialistResult(
            SPECIALIST_ID,
            {
                **shared,
                "status": "abstain",
                "reason": "not_a_normal_map",
                "converted": None,
                "warnings": ["not_a_normal_map"],
            },
            0.0, True, AuthorityMode.CONSULTATIVE,
            notes=(
                "The vectors are not unit length, so the encoding is not a normal map.",
                "Guessing a convention for an unknown encoding would be worse than refusing.",
            ),
        )
    warnings: list[str] = []
    if report["verdict"] == "object-space":
        warnings.append("object_space_input")
    if report["verdict"] == "inverted-z":
        warnings.append("inverted_z_input")
    detection = report["y_convention_detection"]
    if (
        detection["status"] == "decided"
        and target is not None
        and target != detection["convention"]
    ):
        warnings.append("converting_against_the_detected_convention")
    if target is None:
        status = "inspected"
        converted = None
    elif target == declaration:
        status = "unchanged"
        converted = [[pixel for pixel in row] for row in image]
    else:
        status = "converted"
        converted = flip_normal_convention(image)
    confidence = CONFIDENCE_BY_VERDICT.get(report["verdict"], 0.5)
    if detection["status"] == "abstain":
        confidence = min(confidence, 0.7)
    return SpecialistResult(
        SPECIALIST_ID,
        {
            **shared,
            "status": status,
            "reason": None,
            "converted": converted,
            "warnings": warnings,
        },
        round(confidence, 4), False, AuthorityMode.CONSULTATIVE,
        notes=(
            "The conversion follows the declared target; the detection is evidence beside it.",
            "Integrability decides the handedness where the surface bends, and stays silent where it does not.",
        ),
    )
