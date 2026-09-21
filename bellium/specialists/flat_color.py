"""Fill a selected hole with a median colour only when the field is flat."""

from __future__ import annotations

from typing import Any

from bellium.contracts.schema import AuthorityMode, SpecialistResult
from bellium.knn._image import copy_image, shape, validate_mask
from bellium.knn.flat_color import classify_flat

SPECIALIST_ID = "bellium/hybrid/flat-color-helper:v0"


def _median_channel(values: list[int]) -> int:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return int(round((ordered[mid - 1] + ordered[mid]) / 2))


def _median_pixel(pixels: list[tuple[int, int, int]]) -> tuple[int, int, int]:
    return (
        _median_channel([p[0] for p in pixels]),
        _median_channel([p[1] for p in pixels]),
        _median_channel([p[2] for p in pixels]),
    )


def flatten_region(query: dict[str, Any]) -> SpecialistResult:
    image = query.get("image")
    mask = query.get("mask")
    if image is None or mask is None:
        raise ValueError("query needs image and mask")
    validate_mask(image, mask)
    height, width = shape(image)
    hole = [(r, c) for r in range(height) for c in range(width) if mask[r][c]]
    known = [image[r][c] for r in range(height) for c in range(width) if not mask[r][c]]
    if not hole:
        return SpecialistResult(
            SPECIALIST_ID,
            {"status": "safe", "filled": 0, "image": copy_image(image), "certified": False},
            1.0, False, AuthorityMode.CONSULTATIVE,
            notes=("Nothing to fill.",),
        )
    if len(known) < 4:
        return SpecialistResult(
            SPECIALIST_ID,
            {"status": "abstain", "reason": "not_enough_known_pixels", "filled": 0},
            0.0, True, AuthorityMode.CONSULTATIVE,
        )
    verdict = classify_flat({"image": image, "mask": mask})
    if verdict.abstained or verdict.output.get("label") != "flat":
        return SpecialistResult(
            SPECIALIST_ID,
            {"status": "abstain", "reason": "region_not_flat", "knn": verdict.output, "filled": 0},
            0.0, True, AuthorityMode.CONSULTATIVE,
            notes=("Textured fields are not flattened. Escalate to inpaint if needed.",),
        )
    fill = _median_pixel(known)
    painted = copy_image(image)
    for r, c in hole:
        painted[r][c] = fill
    return SpecialistResult(
        SPECIALIST_ID,
        {
            "status": "suggest",
            "filled": len(hole),
            "color": fill,
            "image": painted,
            "knn": verdict.output,
            "certified": False,
        },
        verdict.confidence,
        False, AuthorityMode.CONSULTATIVE,
        notes=("Median fill of a flat field only. Not shading and not inpainting.",),
    )

