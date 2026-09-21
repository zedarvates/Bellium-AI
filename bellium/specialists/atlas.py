"""Bounded atlas packing: a placement plan the caller owns.

The specialist places declared frames on bounded pages and reports the geometry.
It writes no pixels, never scales or rotates a frame, and abstains when the
declared pages cannot hold everything instead of shrinking the assets.
"""

from __future__ import annotations

from typing import Any

from bellium.atlas.packing import (
    check_plan,
    pack,
    parse_frames,
    power_of_two_ceiling,
)
from bellium.contracts.schema import AuthorityMode, SpecialistResult

SPECIALIST_ID = "bellium/hybrid/atlas-packing:v0"
QUERY_KEYS = ("frames", "width", "height", "method", "padding", "max_pages", "power_of_two")
READY_CONFIDENCE = 0.9


def pack_atlas(query: dict[str, Any]) -> SpecialistResult:
    if not isinstance(query, dict):
        raise ValueError("query must be an object")
    unknown = sorted(set(query) - set(QUERY_KEYS))
    if unknown:
        raise ValueError(f"unknown query keys refuse to be ignored: {', '.join(unknown)}")
    frames = parse_frames(query.get("frames"))
    power_of_two = query.get("power_of_two", False)
    if not isinstance(power_of_two, bool):
        raise ValueError("power_of_two must be a boolean")
    width = query.get("width")
    height = query.get("height")
    page_width = power_of_two_ceiling(width) if power_of_two else width
    page_height = power_of_two_ceiling(height) if power_of_two else height
    plan = pack(
        frames,
        width=page_width,
        height=page_height,
        method=query.get("method", "skyline"),
        padding=query.get("padding", 0),
        max_pages=query.get("max_pages", 1),
    )
    violations = check_plan(frames, plan)
    shared = {
        **plan.to_dict(),
        "frame_count": len(frames),
        "unplaced_count": len(plan.unplaced),
        "pixels_written": False,
        "certified": False,
    }
    if violations:
        return SpecialistResult(
            SPECIALIST_ID,
            {**shared, "status": "abstain", "reason": "plan_failed_verification",
             "violations": violations},
            0.0, True, AuthorityMode.CONSULTATIVE,
            notes=("A plan that fails its own invariants is reported, never returned as usable.",),
        )
    if not plan.complete:
        return SpecialistResult(
            SPECIALIST_ID,
            {**shared, "status": "abstain", "reason": "frames_do_not_fit"},
            0.0, True, AuthorityMode.CONSULTATIVE,
            notes=(
                "Frames that do not fit are reported unplaced; they are never shrunk, "
                "cropped or rotated.",
            ),
        )
    return SpecialistResult(
        SPECIALIST_ID,
        {**shared, "status": "ready", "reason": None},
        READY_CONFIDENCE,
        False,
        AuthorityMode.CONSULTATIVE,
        notes=(
            "Placement geometry only; no pixels are written and nothing is imported.",
            "The caller still owns rendering, duplicates and engine conventions.",
        ),
    )

