"""Deterministic sprite-sheet geometry: grid, frames and motion deltas.

Pure measurement helpers. A grid is accepted only when the content runs and the
repeat period agree, or when the caller declares a cell size that the measured
content respects. No learned parameter decides anything here.
"""

from __future__ import annotations

import math
from collections import Counter

from bellium.knn._image import Image, Mask, Rgb, shape
from bellium.knn._texture import estimate_period

MIN_CELL = 4
MIN_FRAMES = 2
DEFAULT_TOLERANCE = 8
PERIOD_STRENGTH = 0.5
BORDER_BAND = 2


def _distance(left: Rgb, right: Rgb) -> float:
    return math.sqrt(sum((left[i] - right[i]) ** 2 for i in range(3)))


def border_samples(image: Image, band: int = BORDER_BAND) -> list[Rgb]:
    height, width = shape(image)
    if height < 2 * band or width < 2 * band:
        raise ValueError("image is too small for a border band")
    return [
        image[r][c]
        for r in range(height)
        for c in range(width)
        if r < band or c < band or r >= height - band or c >= width - band
    ]


def border_background(image: Image, *, band: int = BORDER_BAND) -> tuple[Rgb, float]:
    """Most common border colour plus its disagreement as a 0..1 ratio."""
    samples = border_samples(image, band)
    counts = Counter(tuple(pixel) for pixel in samples)
    color, hits = counts.most_common(1)[0]
    return (color[0], color[1], color[2]), 1.0 - (hits / len(samples))


def foreground_mask(image: Image, *, background: Rgb, tolerance: float = DEFAULT_TOLERANCE) -> Mask:
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)):
        raise ValueError("tolerance must be numeric")
    if not math.isfinite(float(tolerance)) or float(tolerance) < 0:
        raise ValueError("tolerance must be positive and finite")
    height, width = shape(image)
    return [
        [1 if _distance(image[r][c], background) > tolerance else 0 for c in range(width)]
        for r in range(height)
    ]


def presence(mask: Mask, axis: str) -> list[float]:
    """1.0 where the axis line holds at least one foreground pixel."""
    if axis not in ("x", "y"):
        raise ValueError("axis must be x or y")
    height = len(mask)
    width = len(mask[0])
    if axis == "x":
        return [1.0 if any(mask[r][c] for r in range(height)) else 0.0 for c in range(width)]
    return [1.0 if any(row) else 0.0 for row in mask]


def content_runs(profile: list[float]) -> list[tuple[int, int]]:
    """Maximal runs of foreground presence as inclusive (start, end) pairs."""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(profile):
        if value > 0.0:
            if start is None:
                start = index
        elif start is not None:
            runs.append((start, index - 1))
            start = None
    if start is not None:
        runs.append((start, len(profile) - 1))
    return runs


def _runs_cells(size: int, runs: list[tuple[int, int]]) -> tuple[int | None, str | None]:
    """Cell count inferred from one content run per cell."""
    if len(runs) < 2:
        return None, "too_few_content_runs"
    if size % len(runs) != 0:
        return None, "run_count_does_not_divide"
    cell = size // len(runs)
    if cell < MIN_CELL:
        return None, "grid_too_small"
    for start, end in runs:
        if end >= (start // cell + 1) * cell:
            return None, "content_run_crosses_a_cell"
    return len(runs), None


def cell_count(size: int, period: int | None, strength: float) -> tuple[int, int | None]:
    """Cells along one axis from the repeat period of the presence profile."""
    if period is None or strength < PERIOD_STRENGTH:
        return 1, None
    if period < MIN_CELL:
        return 1, "grid_too_small"
    if size % period != 0:
        return 1, "period_does_not_divide"
    return size // period, None


def declared_cells(width: int, height: int, cell_size: tuple[int, int] | None) -> dict[str, int] | None:
    """Verified caller-declared cell counts, or None when nothing was declared."""
    if cell_size is None:
        return None
    if not isinstance(cell_size, (list, tuple)) or len(cell_size) != 2:
        raise ValueError("cell_size must list width and height")
    cells: dict[str, int] = {}
    for axis, size, raw in (("x", width, cell_size[0]), ("y", height, cell_size[1])):
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < MIN_CELL:
            raise ValueError(f"cell_size along {axis} must be an integer of at least {MIN_CELL} pixels")
        if size % raw != 0:
            raise ValueError(f"the declared cell size does not divide the sheet along {axis}")
        cells[axis] = size // raw
    return cells


def _axis_grid(size: int, mask: Mask, axis: str, declared: int | None) -> dict:
    profile = presence(mask, axis)
    runs = content_runs(profile)
    runs_cells, runs_note = _runs_cells(size, runs)
    period, strength = estimate_period(profile, min_period=MIN_CELL)
    period_cells, period_note = cell_count(size, period, strength)
    if period_cells is not None and period_cells < 2:
        period_cells = None
    entry: dict = {
        "size": size,
        "period": period,
        "period_cells": period_cells,
        "strength": round(min(max(strength, 0.0), 1.0), 6),
        "run_count": len(runs),
        "runs": [[start, end] for start, end in runs],
    }
    if declared is not None:
        cell = size // declared
        fits = all(end < (start // cell + 1) * cell for start, end in runs)
        conflict = (runs_cells is not None and runs_cells != declared) or (
            runs and not fits
        )
        entry.update(
            cells=declared,
            refusal="declared_grid_conflicts_with_content" if conflict else None,
            source="declared",
        )
        return entry
    if runs_cells is not None and period_cells is not None and runs_cells != period_cells:
        entry.update(cells=runs_cells, refusal="grid_signals_disagree", source="measured")
    elif runs_cells is not None:
        entry.update(cells=runs_cells, refusal=None, source="measured")
    elif period_cells is not None:
        entry.update(cells=period_cells, refusal=None, source="measured")
    elif runs_note == "too_few_content_runs":
        # A single contiguous run is the only measurable structure: one cell on
        # this axis. The caller can still declare a cell size to confirm it.
        entry.update(cells=1, refusal=None, source="measured", single_run=True)
    else:
        entry.update(cells=1, refusal=runs_note or period_note or "grid_not_confirmed",
                     source="measured")
    return entry


def detect_grid(
    image: Image,
    *,
    background: Rgb | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
    cell_size: tuple[int, int] | None = None,
) -> dict:
    """Uniform frame grid, or an explicit refusal with the measured evidence."""
    height, width = shape(image)
    if background is None:
        color, disagreement = border_background(image)
    else:
        color = (int(background[0]), int(background[1]), int(background[2]))
        samples = border_samples(image)
        disagreement = sum(1 for pixel in samples if _distance(pixel, color) > tolerance) / len(samples)
    mask = foreground_mask(image, background=color, tolerance=tolerance)
    declared = declared_cells(width, height, cell_size)
    axes = {
        axis: _axis_grid(size, mask, axis, None if declared is None else declared[axis])
        for axis, size in (("x", width), ("y", height))
    }
    return {
        "background": color,
        "border_disagreement": round(disagreement, 6),
        "tolerance": float(tolerance),
        "grid_source": "declared" if declared is not None else "measured",
        "axes": axes,
        "mask": mask,
    }


def frame_boxes(grid: dict) -> list[dict]:
    """Uniform cells in reading order. Refuses an unconfirmed grid."""
    axes = grid["axes"]
    refusals = sorted({axes[axis]["refusal"] for axis in ("x", "y") if axes[axis]["refusal"]})
    if refusals:
        raise ValueError(f"grid not confirmed: {', '.join(refusals)}")
    columns = axes["x"]["cells"]
    rows = axes["y"]["cells"]
    if columns * rows < MIN_FRAMES:
        raise ValueError("a sheet needs at least two frames")
    cell_w = axes["x"]["size"] // columns
    cell_h = axes["y"]["size"] // rows
    if cell_w < MIN_CELL or cell_h < MIN_CELL:
        raise ValueError("frames are smaller than the minimum cell")
    boxes = []
    for row in range(rows):
        for column in range(columns):
            boxes.append({
                "index": len(boxes),
                "row": row,
                "column": column,
                "x": column * cell_w,
                "y": row * cell_h,
                "w": cell_w,
                "h": cell_h,
            })
    return boxes


def cell_mask(mask: Mask, box: dict) -> Mask:
    return [row[box["x"]:box["x"] + box["w"]] for row in mask[box["y"]:box["y"] + box["h"]]]


def cell_image(image: Image, box: dict) -> Image:
    return [row[box["x"]:box["x"] + box["w"]] for row in image[box["y"]:box["y"] + box["h"]]]


def content_box(mask: Mask) -> tuple[int, int, int, int] | None:
    rows = [r for r, row in enumerate(mask) if any(row)]
    columns = [c for c in range(len(mask[0])) if any(row[c] for row in mask)]
    if not rows or not columns:
        return None
    return min(columns), min(rows), max(columns), max(rows)


def _centroid(mask: Mask) -> tuple[float, float]:
    height = len(mask)
    width = len(mask[0])
    total = sum(sum(row) for row in mask)
    if total == 0:
        return 0.0, 0.0
    x = sum(c for r in range(height) for c in range(width) if mask[r][c]) / total
    y = sum(r for r in range(height) for c in range(width) if mask[r][c]) / total
    return x, y


def _mean_color(image: Image, mask: Mask) -> Rgb | None:
    pixels = [
        image[r][c]
        for r in range(len(mask))
        for c in range(len(mask[0]))
        if mask[r][c]
    ]
    if not pixels:
        return None
    return (
        round(sum(pixel[0] for pixel in pixels) / len(pixels)),
        round(sum(pixel[1] for pixel in pixels) / len(pixels)),
        round(sum(pixel[2] for pixel in pixels) / len(pixels)),
    )


def _mask_delta(previous: Mask, current: Mask) -> tuple[int, int, int]:
    height = len(previous)
    width = len(previous[0])
    union = 0
    added = 0
    removed = 0
    for r in range(height):
        for c in range(width):
            a = previous[r][c]
            b = current[r][c]
            if a or b:
                union += 1
            if b and not a:
                added += 1
            if a and not b:
                removed += 1
    return union, added, removed


def loop_features(first: tuple[Mask, Image], last: tuple[Mask, Image]) -> dict[str, float]:
    """Compact first/last comparison in [0, 1]. High values mean a drifting loop."""
    first_mask, first_image = first
    last_mask, last_image = last
    height = len(first_mask)
    width = len(first_mask[0])
    if len(last_mask) != height or len(last_mask[0]) != width:
        raise ValueError("compared frames must share a cell size")
    diagonal = math.hypot(width, height) or 1.0
    union, added, removed = _mask_delta(first_mask, last_mask)
    if union == 0:
        return {
            "iou_mismatch": 0.0,
            "added_ratio": 0.0,
            "removed_ratio": 0.0,
            "centroid_shift": 0.0,
            "area_ratio": 0.0,
            "color_delta": 0.0,
            "bbox_shift": 0.0,
        }
    first_area = sum(sum(row) for row in first_mask)
    last_area = sum(sum(row) for row in last_mask)
    shared = union - added - removed
    centroid_first = _centroid(first_mask)
    centroid_last = _centroid(last_mask)
    shift = math.hypot(centroid_last[0] - centroid_first[0], centroid_last[1] - centroid_first[1])
    first_box = content_box(first_mask)
    last_box = content_box(last_mask)
    bbox_shift = 0.0
    if first_box and last_box:
        bbox_shift = max(
            abs(last_box[0] - first_box[0]),
            abs(last_box[1] - first_box[1]),
        ) / max(width, height)
    first_color = _mean_color(first_image, first_mask)
    last_color = _mean_color(last_image, last_mask)
    color_delta = 0.0
    if first_color and last_color:
        color_delta = _distance(first_color, last_color) / (255.0 * math.sqrt(3))
    smaller = min(first_area, last_area)
    bigger = max(first_area, last_area)
    return {
        "iou_mismatch": round(min(1.0, 1.0 - (shared / union)), 6),
        "added_ratio": round(min(1.0, added / union), 6),
        "removed_ratio": round(min(1.0, removed / union), 6),
        "centroid_shift": round(min(1.0, shift / diagonal), 6),
        "area_ratio": round(0.0 if bigger == 0 else min(1.0, 1.0 - (smaller / bigger)), 6),
        "color_delta": round(min(1.0, color_delta), 6),
        "bbox_shift": round(min(1.0, bbox_shift), 6),
    }


def delta_features(previous: Mask, current: Mask, position: float,
                   previous_change: float | None = None) -> dict[str, float]:
    """Motion delta between two consecutive frames in [0, 1]."""
    if position < 0.0 or position > 1.0:
        raise ValueError("position must be between 0 and 1")
    height = len(previous)
    width = len(previous[0])
    area = width * height
    _union, added, removed = _mask_delta(previous, current)
    changed_ratio = (added + removed) / area
    previous_area = sum(sum(row) for row in previous)
    current_area = sum(sum(row) for row in current)
    bigger = max(previous_area, current_area)
    centroid_previous = _centroid(previous)
    centroid_current = _centroid(current)
    shift = math.hypot(
        centroid_current[0] - centroid_previous[0],
        centroid_current[1] - centroid_previous[1],
    )
    if previous_change is None:
        slope = 0.5
    else:
        span = max(changed_ratio, previous_change, 0.02)
        slope = ((changed_ratio - previous_change) / span) / 2.0 + 0.5
    return {
        "changed_ratio": round(min(1.0, changed_ratio), 6),
        "added_ratio": round(0.0 if area == 0 else min(1.0, added / area), 6),
        "removed_ratio": round(0.0 if area == 0 else min(1.0, removed / area), 6),
        "centroid_shift": round(min(1.0, shift / (math.hypot(width, height) or 1.0)), 6),
        "area_delta": round(
            0.0 if bigger == 0 else min(1.0, abs(current_area - previous_area) / bigger), 6
        ),
        "change_slope": round(min(1.0, max(0.0, slope)), 6),
        "position": round(position, 6),
    }


def deterministic_loop_verdict(features: dict[str, float], *, mismatch_limit: float = 0.18,
                               shift_limit: float = 0.06) -> str:
    """Published baseline kept beside the k-NN tier."""
    mismatch = float(features["iou_mismatch"])
    shift = float(features["centroid_shift"])
    bbox = float(features["bbox_shift"])
    loops = mismatch <= mismatch_limit and shift <= shift_limit and bbox <= shift_limit
    return "loops" if loops else "drifts"


def deterministic_phase_verdict(features: dict[str, float], *, hold_limit: float = 0.03,
                               impact_limit: float = 0.30) -> str:
    """Published baseline for the animation phase, kept beside the nano tier."""
    change = float(features["changed_ratio"])
    slope = float(features["change_slope"])
    if change < hold_limit:
        return "hold"
    if slope > 0.62:
        return "build"
    if slope < 0.38:
        return "recover"
    return "impact" if change >= impact_limit else "hold"
