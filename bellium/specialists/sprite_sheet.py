"""Sprite-sheet preparation: measured grid, frame phases and a loop check.

The grid must be confirmed or declared, every frame stays untouched, and the
result is a report with frame boxes. Pixels are only returned on request.
"""

from __future__ import annotations

from typing import Any

from bellium.contracts.schema import AuthorityMode, SpecialistResult
from bellium.knn._image import shape
from bellium.knn._sheet import (
    cell_image,
    cell_mask,
    content_box,
    delta_features,
    detect_grid,
    frame_boxes,
)
from bellium.knn.clip_loop import FAMILIES, classify_loop
from bellium.knn._sheet import loop_features
from bellium.nano_nn.frame_phase import classify_phase
from bellium.specialists.animation_timing import report_timing

SPECIALIST_ID = "bellium/hybrid/sprite-sheet-prep:v0"
DEFAULT_MAX_FRAMES = 256
MIXED_BACKGROUND = 0.2


def _coverage(mask: list[list[int]]) -> float:
    area = len(mask) * len(mask[0])
    if area == 0:
        return 0.0
    return round(sum(sum(row) for row in mask) / area, 6)


def prepare_sprite_sheet(query: dict[str, Any]) -> SpecialistResult:
    image = query.get("image")
    if image is None:
        raise ValueError("query needs image")
    shape(image)
    family = str(query.get("family") or "").strip()
    if family not in FAMILIES:
        return SpecialistResult(
            SPECIALIST_ID,
            {"status": "abstain", "reason": "unknown_family", "family": family, "frames": []},
            0.0, True, AuthorityMode.CONSULTATIVE,
            notes=("Unknown clip family; no frame grid is invented.",),
        )
    maximum = query.get("max_frames", DEFAULT_MAX_FRAMES)
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 2:
        raise ValueError("max_frames must be an integer of at least two")
    grid = detect_grid(
        image,
        background=query.get("background"),
        tolerance=query.get("tolerance", 8),
        cell_size=query.get("cell_size"),
    )
    evidence = {
        axis: {key: value for key, value in grid["axes"][axis].items() if key != "runs"}
        for axis in ("x", "y")
    }
    warnings: list[str] = []
    if grid["border_disagreement"] > MIXED_BACKGROUND:
        warnings.append("mixed_background")
    for axis in ("x", "y"):
        if grid["axes"][axis].get("single_run") and grid["grid_source"] == "measured":
            warnings.append(f"single_{'row' if axis == 'y' else 'column'}_assumed")
    try:
        boxes = frame_boxes(grid)
    except ValueError as error:
        return SpecialistResult(
            SPECIALIST_ID,
            {
                "status": "abstain",
                "reason": "grid_not_confirmed",
                "family": family,
                "detail": str(error),
                "background": grid["background"],
                "border_disagreement": grid["border_disagreement"],
                "grid_source": grid["grid_source"],
                "axes": evidence,
                "frames": [],
                "source_preserved": True,
            },
            0.0, True, AuthorityMode.CONSULTATIVE,
            notes=("The sheet was not split: the grid could not be confirmed.",),
        )
    if len(boxes) > maximum:
        return SpecialistResult(
            SPECIALIST_ID,
            {"status": "abstain", "reason": "too_many_frames", "family": family,
             "frame_count": len(boxes), "limit": maximum, "frames": []},
            0.0, True, AuthorityMode.CONSULTATIVE,
        )
    cells = [
        (box, cell_mask(grid["mask"], box), cell_image(image, box))
        for box in boxes
    ]
    empty = [box["index"] for box, mask, _ in cells if not any(any(row) for row in mask)]
    if len(empty) == len(cells):
        return SpecialistResult(
            SPECIALIST_ID,
            {"status": "abstain", "reason": "no_foreground", "family": family, "frames": []},
            0.0, True, AuthorityMode.CONSULTATIVE,
        )
    if empty:
        warnings.append("empty_frames")
    index_range = len(cells) - 1
    frames = []
    previous_change: float | None = None
    changes: list[float] = []
    phases: list[str | None] = []
    for index, (box, mask, cell) in enumerate(cells):
        entry: dict[str, Any] = {
            "index": index,
            "row": box["row"],
            "column": box["column"],
            "box": {"x": box["x"], "y": box["y"], "w": box["w"], "h": box["h"]},
            "coverage": _coverage(mask),
            "content_box": content_box(mask),
        }
        if index == 0:
            entry["phase"] = "start"
            entry["phase_source"] = "position"
        else:
            position = index / index_range if index_range else 0.0
            delta = delta_features(cells[index - 1][1], mask, position, previous_change)
            verdict = classify_phase(delta)
            entry["delta"] = delta
            entry["phase"] = verdict.output.get("phase")
            entry["phase_baseline"] = verdict.output["baseline"]
            entry["phase_status"] = verdict.output["status"]
            entry["phase_confidence"] = verdict.confidence
            entry["phase_agrees_with_baseline"] = verdict.output.get("agrees_with_baseline")
            previous_change = delta["changed_ratio"]
            changes.append(delta["changed_ratio"])
            phases.append(entry["phase"])
        if query.get("include_pixels") is True:
            entry["image"] = cell
        frames.append(entry)
    if changes:
        peak = max(changes)
        if peak < 0.03:
            warnings.append("no_visible_motion")
        elif all(phase not in ("impact", "build") for phase in phases):
            warnings.append("no_peak_frame")
        if changes.index(peak) == len(changes) - 1:
            warnings.append("peak_at_last_frame")
    _, first_mask, first_cell = cells[0]
    _, last_mask, last_cell = cells[-1]
    loop = classify_loop({
        "family": family,
        "frames": {
            "first": {"image": first_cell, "mask": first_mask},
            "last": {"image": last_cell, "mask": last_mask},
        },
    })
    if loop.abstained:
        warnings.append("loop_not_confirmed")
    elif loop.output["verdict"] == "drifts":
        warnings.append("loop_drifts")
    series = [entry["delta"]["changed_ratio"] for entry in frames if "delta" in entry]
    mismatches = [
        loop_features(
            (cells[index - 1][1], cells[index - 1][2]),
            (cells[index][1], cells[index][2]),
        )["iou_mismatch"]
        for index in range(1, len(cells))
    ]
    timing = report_timing(
        {"family": family, "series": series, "frame_mismatches": mismatches}
    )
    for warning in timing.output.get("warnings", []):
        if warning not in warnings:
            warnings.append(warning)
    confidence = 0.4 if warnings else 0.75
    return SpecialistResult(
        SPECIALIST_ID,
        {
            "status": "ready",
            "family": family,
            "grid_source": grid["grid_source"],
            "background": grid["background"],
            "border_disagreement": grid["border_disagreement"],
            "cell": {"w": boxes[0]["w"], "h": boxes[0]["h"]},
            "frame_count": len(frames),
            "frames": frames,
            "empty_frames": empty,
            "loop": loop.output,
            "timing": timing.output,
            "warnings": warnings,
            "certified": False,
            "source_preserved": True,
        },
        confidence,
        False, AuthorityMode.CONSULTATIVE,
        notes=(
            "Geometry and timing report only. No export, no engine import, no rig.",
            "Phases describe motion change, not semantic animation labels.",
        ),
    )
