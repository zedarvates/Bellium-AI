"""Direct filters and a bounded magic eraser.

The filter specialist replays a declared recipe on the source and returns the
result; the eraser reuses the existing inpaint router and patch k-NN, so an
uncertain selection abstains or asks for escalation instead of inventing content.
"""

from __future__ import annotations

from typing import Any

from bellium.contracts.schema import AuthorityMode, SpecialistResult
from bellium.editing.stack import EditStack, EditStep
from bellium.knn._image import Image, Mask, shape, validate_mask
from bellium.knn.patch_inpaint import inpaint
from bellium.specialists.inpaint_router import route_inpaint

SPECIALIST_ID = "bellium/hybrid/direct-filters:v0"
ERASER_ID = "bellium/hybrid/magic-eraser:v0"
MASK_THRESHOLD = 0.5


def _steps(raw: object) -> list[EditStep]:
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError("a recipe must list at least one step")
    steps = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"step {index} must be an object")
        steps.append(EditStep(
            name=str(item.get("name") or ""),
            parameters=item.get("parameters") or {},
            mask=item.get("mask"),
            label=str(item.get("label") or ""),
        ))
    return steps


def edit_image(query: dict[str, Any]) -> SpecialistResult:
    image = query.get("image")
    if image is None:
        raise ValueError("query needs image")
    shape(image)
    steps = _steps(query.get("recipe"))
    stack = EditStack(image)
    for step in steps:
        stack.add(step)
    edited = stack.export()
    return SpecialistResult(
        SPECIALIST_ID,
        {
            "status": "ready",
            "image": edited,
            "recipe": stack.to_recipe(),
            "steps": stack.history(),
            "source_preserved": True,
            "certified": False,
        },
        0.9,
        False,
        AuthorityMode.CONSULTATIVE,
        notes=(
            "Filtered preview built from the declared recipe; the source is untouched.",
            "Colours are clipped in the source space; this is not a colour-managed export.",
        ),
    )


def _binary_mask(mask: object) -> Mask:
    if not isinstance(mask, list) or not mask:
        raise ValueError("mask is empty")
    binary: Mask = []
    for row in mask:
        if not isinstance(row, list) or not row:
            raise ValueError("mask rows must be non-empty lists")
        line = []
        for value in row:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("mask values must be numeric")
            number = float(value)
            if not 0.0 <= number <= 1.0:
                raise ValueError("mask values must be between 0 and 1")
            line.append(1 if number >= MASK_THRESHOLD else 0)
        binary.append(line)
    return binary


def erase_region(query: dict[str, Any]) -> SpecialistResult:
    """Propose a bounded fill for a selected region, or refuse and escalate."""
    image: Image | None = query.get("image")
    if image is None:
        raise ValueError("query needs image")
    shape(image)
    mask = _binary_mask(query.get("mask"))
    validate_mask(image, mask)
    if not any(any(row) for row in mask):
        return SpecialistResult(
            ERASER_ID,
            {"status": "abstain", "reason": "empty_selection", "image": None,
             "source_preserved": True, "certified": False},
            0.0, True, AuthorityMode.CONSULTATIVE,
            notes=("Nothing is selected, so nothing is reconstructed.",),
        )
    router = route_inpaint(image, mask)
    label = router.output.get("label")
    if label != "patch_knn":
        return SpecialistResult(
            ERASER_ID,
            {"status": "abstain",
             "reason": "escalate" if label == "escalate" else "router_abstained",
             "router": router.output, "image": None,
             "source_preserved": True, "certified": False},
            router.confidence or 0.0, True, AuthorityMode.CONSULTATIVE,
            notes=(
                "The inpaint router did not confirm the bounded patch method.",
                "An uncertain fill stays a suggestion to escalate, never a silent edit.",
            ),
        )
    candidate = inpaint(image, mask)
    if candidate.abstained:
        return SpecialistResult(
            ERASER_ID,
            {"status": "abstain", "reason": "candidate_not_confirmed",
             "router": router.output, "candidate": candidate.output, "image": None,
             "source_preserved": True, "certified": False},
            0.0, True, AuthorityMode.CONSULTATIVE,
            notes=(
                "The patch fill could not be confirmed locally; nothing is proposed.",
            ),
        )
    return SpecialistResult(
        ERASER_ID,
        {
            "status": "preview",
            "image": candidate.output.get("image"),
            "router": router.output,
            "candidate": {
                key: value for key, value in candidate.output.items() if key != "image"
            },
            "source_preserved": True,
            "certified": False,
        },
        candidate.confidence,
        False,
        AuthorityMode.CONSULTATIVE,
        notes=(
            "Preview only: the fill is a bounded patch candidate, not a reconstruction.",
            "Compare it before applying; the source image is untouched.",
        ),
    )
