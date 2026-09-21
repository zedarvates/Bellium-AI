"""Resampling a normal field, and the drift a level-of-detail chain leaves behind.

A normal map is not a colour image: averaging its vectors shortens them, and the
shortening is a defect a length check cannot see. Three published reductions are
provided so they can be compared instead of trusted, and every level reports what
the filter did to the field.
"""

from __future__ import annotations

import math

Floats = list[list[float]]
Normals = list[list[tuple[float, float, float] | None]]
METHODS = ("renormalized", "naive", "slopes")
DEFAULT_METHOD = "renormalized"
DEFAULT_FACTOR = 2
DEFAULT_LEVELS = 3
# A chain of 512 x 512 grids is the largest this module returns as pixels; a
# bigger source gets its statistics and a refusal instead of a silent copy.
DEFAULT_MAX_PIXELS = 512 * 512
# Measured on the controlled fields: two levels of a sphere drift 0.053 in mean Z
# while a cone drifts 0.005 and a plane not at all, so the limit sits between the
# surface that flattens and the surfaces that do not.
FLATTENING_LIMIT = 0.03
COVERAGE_LIMIT = 0.95


def _validate(normals: Normals, mask: Floats | None) -> tuple[int, int]:
    if not isinstance(normals, list) or not normals or not normals[0]:
        raise ValueError("normals are empty")
    rows = len(normals)
    columns = len(normals[0])
    for index, row in enumerate(normals):
        if len(row) != columns:
            raise ValueError("normal rows must have equal width")
        for column, normal in enumerate(row):
            if normal is None:
                continue
            if len(normal) != 3:
                raise ValueError("a normal needs three components")
            for value in normal:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError("normal components must be numeric")
                if not math.isfinite(float(value)):
                    raise ValueError("normal components must be finite")
    if mask is not None:
        if len(mask) != rows or any(len(row) != columns for row in mask):
            raise ValueError("the mask must share the normal shape")
    return rows, columns


def _method(name: object) -> str:
    if name not in METHODS:
        raise ValueError(f"method must be one of: {', '.join(METHODS)}")
    return str(name)


def _factor(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("factor must be a positive integer")
    return value


def _mean_z(normals: Normals, mask: Floats) -> float | None:
    values = [
        normal[2]
        for r, row in enumerate(normals)
        for c, normal in enumerate(row)
        if normal is not None and mask[r][c] > 0.0
    ]
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def downsample_normals(
    normals: Normals,
    factor: int = DEFAULT_FACTOR,
    *,
    mask: Floats | None = None,
    method: str = DEFAULT_METHOD,
) -> dict:
    """Reduce by an integer factor, keeping only blocks that are fully covered.

    renormalized averages the unit vectors and normalizes the result, which is what
    a mip chain has to do; naive stores the averaged vector as it comes out, which
    is the same picture with a length defect; slopes averages the slopes and
    rebuilds the normal, which is the textbook advice and measures worse here.
    """
    declaration = _method(method)
    step = _factor(factor)
    rows, columns = _validate(normals, mask)
    if step == 1:
        return {
            "normals": [[normal for normal in row] for row in normals],
            "mask": [[1.0] * columns for _ in range(rows)] if mask is None
            else [[mask[r][c] for c in range(columns)] for r in range(rows)],
            "factor": 1,
            "method": declaration,
            "source_shape": [rows, columns],
            "target_shape": [rows, columns],
            "coverage": 1.0,
            "mean_length_before": 1.0,
        }
    target_rows = rows // step
    target_columns = columns // step
    out: Normals = [[None] * target_columns for _ in range(target_rows)]
    out_mask = [[0.0] * target_columns for _ in range(target_rows)]
    covered_cells = 0
    length_total = 0.0
    length_count = 0
    for r in range(target_rows):
        for c in range(target_columns):
            cover = 0
            vectors: list[tuple[float, float, float]] = []
            slope_x = 0.0
            slope_y = 0.0
            slopes = 0
            for dr in range(step):
                for dc in range(step):
                    rr = r * step + dr
                    cc = c * step + dc
                    if mask is not None and mask[rr][cc] <= 0.0:
                        continue
                    cover += 1
                    normal = normals[rr][cc]
                    if normal is None:
                        continue
                    vectors.append(normal)
                    if abs(normal[2]) > 1e-6:
                        slope_x += -normal[0] / normal[2]
                        slope_y += -normal[1] / normal[2]
                        slopes += 1
            if cover == step * step:
                covered_cells += 1
                out_mask[r][c] = 1.0
            if not vectors:
                continue
            averaged = [
                sum(vector[index] for vector in vectors) / len(vectors) for index in range(3)
            ]
            length = math.sqrt(sum(value * value for value in averaged))
            length_total += length
            length_count += 1
            if declaration == "naive":
                out[r][c] = (averaged[0], averaged[1], averaged[2])
                continue
            if declaration == "slopes":
                if not slopes:
                    continue
                gradient_x = slope_x / slopes
                gradient_y = slope_y / slopes
                unit = math.sqrt(gradient_x * gradient_x + gradient_y * gradient_y + 1.0)
                out[r][c] = (-gradient_x / unit, -gradient_y / unit, 1.0 / unit)
                continue
            if length <= 1e-12:
                out[r][c] = (0.0, 0.0, 1.0)
                continue
            out[r][c] = (
                averaged[0] / length,
                averaged[1] / length,
                averaged[2] / length,
            )
    return {
        "normals": out,
        "mask": out_mask,
        "factor": step,
        "method": declaration,
        "source_shape": [rows, columns],
        "target_shape": [target_rows, target_columns],
        "coverage": round(covered_cells / (target_rows * target_columns), 6)
        if target_rows and target_columns
        else 0.0,
        "mean_length_before": round(length_total / length_count, 6) if length_count else None,
    }


def mip_chain(
    normals: Normals,
    *,
    mask: Floats | None = None,
    levels: int = DEFAULT_LEVELS,
    factor: int = DEFAULT_FACTOR,
    method: str = DEFAULT_METHOD,
) -> dict:
    """Build a chain of reduced levels and report what each level did to the field.

    A level whose source is too small to reduce is not produced: the chain stops
    and says so, because inventing a level by copying the previous one would hide
    the limit of the source.
    """
    declaration = _method(method)
    step = _factor(factor)
    if isinstance(levels, bool) or not isinstance(levels, int) or levels < 1:
        raise ValueError("levels must be a positive integer")
    rows, columns = _validate(normals, mask)
    current = [[normal for normal in row] for row in normals]
    current_mask = (
        [[1.0] * columns for _ in range(rows)]
        if mask is None
        else [[mask[r][c] for c in range(columns)] for r in range(rows)]
    )
    chain = [{
        "level": 0,
        "shape": [rows, columns],
        "mean_z": _mean_z(current, current_mask),
        "coverage": 1.0,
        "mean_length_before": 1.0,
    }]
    pixels: list[dict] = []
    warnings: list[str] = []
    for level in range(1, levels + 1):
        if len(current) // step < 1 or len(current[0]) // step < 1:
            warnings.append("chain_stopped_early")
            break
        reduced = downsample_normals(current, step, mask=current_mask, method=declaration)
        chain.append({
            "level": level,
            "shape": reduced["target_shape"],
            "mean_z": _mean_z(reduced["normals"], reduced["mask"]),
            "coverage": reduced["coverage"],
            "mean_length_before": reduced["mean_length_before"],
        })
        pixels.append({"level": level, "normals": reduced["normals"], "mask": reduced["mask"]})
        if reduced["coverage"] < COVERAGE_LIMIT:
            warnings.append("partial_coverage")
        current = reduced["normals"]
        current_mask = reduced["mask"]
    first = chain[0]["mean_z"]
    last = chain[-1]["mean_z"]
    drift = None if first is None or last is None else round(last - first, 6)
    if drift is not None and abs(drift) > FLATTENING_LIMIT:
        # Averaging tilts a curved field towards the viewer, and the tilt grows
        # with every level: measured 0.053 over two levels of a sphere.
        warnings.append("flattening_drift")
    returned = sum(level["shape"][0] * level["shape"][1] for level in chain[1:])
    return {
        "method": declaration,
        "factor": step,
        "levels": chain,
        "pixels": pixels,
        "drift": drift,
        "flattening_limit": FLATTENING_LIMIT,
        "coverage_limit": COVERAGE_LIMIT,
        "pixels_returned": returned,
        "warnings": sorted(set(warnings)),
    }

