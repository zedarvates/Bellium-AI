"""Surface height from a normal field, up to a constant.

Unit normals determine the gradient of the height, so the height is recoverable
only up to one additive constant: every error here is reported with that offset
removed, and the offset itself is reported separately. Four integrators are
provided so they can be compared instead of trusted, and the measured table
records where each one wins.
"""

from __future__ import annotations

import math

Floats = list[list[float]]
Normals = list[list[tuple[float, float, float] | None]]
METHODS = ("least-squares", "cumulative", "cumulative-average", "cumulative-vertical")
# Measured on the controlled geometries: the row/column average has the lowest
# mean relative error of the three, so it is the default. The other two stay
# available because each wins outright on a specific family of surfaces.
DEFAULT_METHOD = "cumulative-average"
DEFAULT_ITERATIONS = 200
DEFAULT_TOLERANCE = 1e-3
# Measured on the controlled fields. Sampling alone contributes up to 0.125: a
# 32 x 32 sphere of analytic normals measures 0.069 and an analytic wave 0.125,
# because a sampled normal is not exactly the gradient of the sampled height, and
# both of those integrate at 0.06 and 0.08 of relative error. So the bands are
# wide on purpose. A bounded rotation saturates near 0.17 whatever its strength,
# and a slope that alternates every row, which is the smallest field that is not a
# gradient at all, measures 2.0.
CURL_LIMIT = 0.15
CURL_SUSPECT_LIMIT = 0.25


def integrability(slope_x: Floats, slope_y: Floats, valid: Floats) -> dict:
    """How far the slope field is from being the gradient of any height.

    A gradient field has no curl, so the curl measures exactly the part of a field
    that no integrator can turn into a surface. The curl is taken around a cell of
    four pixels and the cell counts only when all four corners are valid, because
    an unpaired difference at the edge of a mask invents curl that is not in the
    field: the first version of this check reported 0.12 on a smooth sphere for
    that reason. A second version used central differences over two pixels and was
    blind to a slope that alternates sign every row, the classic odd-even
    decoupling, which the cell form catches at a ratio of 2.0.

    The ratio is divided by the mean slope magnitude, so it is scale free, but not
    unbiased: sampling a curved analytic surface already costs 0.069 to 0.125 at
    32 x 32, and a bounded rotation saturates near 0.17 whatever its strength
    because the magnitude it is divided by grows with the rotation. Read the bands
    as bands: below 0.15 the field behaves like a gradient, and above 0.25 it does
    not, whatever the cause.

    This is a statement about the field, not about an integrator: the measured
    table shows a perturbed plane with 0.21 of curl integrating at 0.11 of
    relative error while a perturbed wave with 0.34 integrates at 0.27.
    """
    rows = len(valid)
    columns = len(valid[0])
    total_curl = 0.0
    total_magnitude = 0.0
    counted = 0
    for r in range(rows - 1):
        for c in range(columns - 1):
            if not (
                valid[r][c]
                and valid[r + 1][c]
                and valid[r][c + 1]
                and valid[r + 1][c + 1]
            ):
                continue
            curl = (slope_x[r + 1][c] - slope_x[r][c]) - (
                slope_y[r][c + 1] - slope_y[r][c]
            )
            total_curl += abs(curl)
            total_magnitude += math.hypot(slope_x[r][c], slope_y[r][c])
            counted += 1
    if counted == 0:
        return {
            "curl_ratio": None,
            "cells": 0,
            "verdict": "unknown",
            "limit": CURL_LIMIT,
            "suspect_limit": CURL_SUSPECT_LIMIT,
        }
    ratio = (total_curl / counted) / (total_magnitude / counted + 1e-9)
    if ratio <= CURL_LIMIT:
        verdict = "integrable"
    elif ratio <= CURL_SUSPECT_LIMIT:
        verdict = "suspect"
    else:
        verdict = "not_integrable"
    return {
        "curl_ratio": round(ratio, 6),
        "cells": counted,
        "verdict": verdict,
        "limit": CURL_LIMIT,
        "suspect_limit": CURL_SUSPECT_LIMIT,
    }


def _validate(normals: Normals, mask: Floats | None) -> tuple[int, int]:
    if not isinstance(normals, list) or not normals or not normals[0]:
        raise ValueError("normals are empty")
    height = len(normals)
    width = len(normals[0])
    for row in normals:
        if len(row) != width:
            raise ValueError("normal rows must have equal width")
        for normal in row:
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
        if len(mask) != height or any(len(row) != width for row in mask):
            raise ValueError("the mask must share the normal shape")
    return height, width


def gradients(
    normals: Normals,
    mask: Floats | None = None,
    *,
    min_cosine: float = 0.0,
) -> tuple[Floats, Floats, Floats]:
    """Slopes of the height surface and the validity mask, all deterministic.

    A normal that lies close to the image plane gives a slope divided by a
    near-zero vertical component, which is noise rather than geometry.
    min_cosine declares how close is too close: at the default only an exactly
    edge-on normal is refused, and a capture-derived field wants a real floor.
    """
    if isinstance(min_cosine, bool) or not isinstance(min_cosine, (int, float)):
        raise ValueError("min_cosine must be numeric")
    if not math.isfinite(float(min_cosine)) or not 0.0 <= float(min_cosine) < 1.0:
        raise ValueError("min_cosine must be between 0 and 1")
    floor = float(min_cosine)
    height, width = _validate(normals, mask)
    slope_x = [[0.0] * width for _ in range(height)]
    slope_y = [[0.0] * width for _ in range(height)]
    valid = [[0.0] * width for _ in range(height)]
    for r in range(height):
        for c in range(width):
            normal = normals[r][c]
            if normal is None or (mask is not None and mask[r][c] <= 0.0):
                continue
            if abs(normal[2]) < max(floor, 1e-6):
                # A surface seen exactly edge-on has no recoverable slope.
                continue
            slope_x[r][c] = -normal[0] / normal[2]
            slope_y[r][c] = -normal[1] / normal[2]
            valid[r][c] = 1.0
    return slope_x, slope_y, valid


def _along_rows(slope_x: Floats, valid: Floats) -> Floats:
    height = len(slope_x)
    width = len(slope_x[0])
    along_rows = [[0.0] * width for _ in range(height)]
    for r in range(height):
        total = 0.0
        for c in range(1, width):
            if valid[r][c] and valid[r][c - 1]:
                total += slope_x[r][c]
            along_rows[r][c] = total
    return along_rows


def _along_columns(slope_y: Floats, valid: Floats) -> Floats:
    height = len(slope_y)
    width = len(slope_y[0])
    along_columns = [[0.0] * width for _ in range(height)]
    for c in range(width):
        total = 0.0
        for r in range(1, height):
            if valid[r][c] and valid[r - 1][c]:
                total += slope_y[r][c]
            along_columns[r][c] = total
    return along_columns


def integrate_cumulative(slope_x: Floats, slope_y: Floats, valid: Floats) -> Floats:
    """Integrate along rows, then align the rows on the difference of the integrals.

    The alignment compares two row integrals column by column, so a surface whose
    rows are identical, such as a tilted plane, carries no vertical signal there
    and that slope is dropped: the measured table records the loss. The vertical
    and least-squares integrators restore it from the column slopes instead.
    """
    along_rows = _along_rows(slope_x, valid)
    height = len(along_rows)
    width = len(along_rows[0])
    offsets = [0.0] * height
    for r in range(1, height):
        differences = [
            along_rows[r][c] - along_rows[r - 1][c]
            for c in range(width)
            if valid[r][c] and valid[r - 1][c]
        ]
        step = sum(differences) / len(differences) if differences else 0.0
        offsets[r] = offsets[r - 1] + step
    out = [[0.0] * width for _ in range(height)]
    for r in range(height):
        for c in range(width):
            if not valid[r][c]:
                continue
            out[r][c] = along_rows[r][c] + offsets[r]
    return out


def integrate_cumulative_vertical(slope_x: Floats, slope_y: Floats, valid: Floats) -> Floats:
    """Integrate along rows, then align the rows on the mean column slope.

    Exact on a constant slope, which least-squares also achieves at roughly two
    hundred times the compute, and clearly better than the row-difference
    alignment on an oscillating surface. On a closed silhouette the row-difference
    alignment wins instead: both numbers are in the measured table.
    """
    along_rows = _along_rows(slope_x, valid)
    height = len(along_rows)
    width = len(along_rows[0])
    offsets = [0.0] * height
    for r in range(1, height):
        steps = [
            slope_y[r][c] for c in range(width) if valid[r][c] and valid[r - 1][c]
        ]
        step = sum(steps) / len(steps) if steps else 0.0
        offsets[r] = offsets[r - 1] + step
    return [
        [along_rows[r][c] + offsets[r] if valid[r][c] else 0.0 for c in range(width)]
        for r in range(height)
    ]


def integrate_least_squares(
    slope_x: Floats,
    slope_y: Floats,
    valid: Floats,
    *,
    iterations: int = DEFAULT_ITERATIONS,
    tolerance: float = DEFAULT_TOLERANCE,
) -> tuple[Floats, dict]:
    """Gauss-Seidel solve of the Poisson equation built from the slopes.

    Deterministic: a fixed sweep count, in a fixed order, from a zero start.
    Gauss-Seidel removes detail-scale error quickly and large-scale curvature
    slowly, so the sweep count is reported together with the residual of the last
    sweep instead of being presented as a convergence guarantee.
    """
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise ValueError("iterations must be a positive integer")
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)):
        raise ValueError("tolerance must be numeric")
    if not math.isfinite(float(tolerance)) or float(tolerance) <= 0.0:
        raise ValueError("tolerance must be positive and finite")
    height = len(slope_x)
    width = len(slope_x[0])
    divergence = [[0.0] * width for _ in range(height)]
    for r in range(height):
        for c in range(width):
            if not valid[r][c]:
                continue
            dx = 0.0
            if c + 1 < width and valid[r][c + 1]:
                dx += slope_x[r][c + 1] - slope_x[r][c]
            dy = 0.0
            if r + 1 < height and valid[r + 1][c]:
                dy += slope_y[r + 1][c] - slope_y[r][c]
            divergence[r][c] = dx + dy
    field = [[0.0] * width for _ in range(height)]
    residual = 0.0
    for _ in range(iterations):
        residual = 0.0
        for r in range(height):
            for c in range(width):
                if not valid[r][c]:
                    continue
                total = 0.0
                neighbours = 0
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < height and 0 <= cc < width and valid[rr][cc]:
                        total += field[rr][cc]
                        neighbours += 1
                if neighbours == 0:
                    continue
                updated = (total - divergence[r][c]) / neighbours
                residual = max(residual, abs(updated - field[r][c]))
                field[r][c] = updated
    # A Poisson solve sees only the curvature of the surface: a constant slope is
    # a solution of the homogeneous equation, so the linear part is restored from
    # the mean gradient instead of being silently dropped.
    mean_slope_x = 0.0
    mean_slope_y = 0.0
    count = 0
    sum_c = 0.0
    sum_r = 0.0
    for r in range(height):
        for c in range(width):
            if not valid[r][c]:
                continue
            mean_slope_x += slope_x[r][c]
            mean_slope_y += slope_y[r][c]
            sum_c += c
            sum_r += r
            count += 1
    if count:
        mean_slope_x /= count
        mean_slope_y /= count
        centre_c = sum_c / count
        centre_r = sum_r / count
        for r in range(height):
            for c in range(width):
                if valid[r][c]:
                    field[r][c] += (
                        mean_slope_x * (c - centre_c) + mean_slope_y * (r - centre_r)
                    )
    return field, {
        "iterations": iterations,
        "last_change": round(residual, 8),
        "tolerance": round(float(tolerance), 8),
        "converged": residual <= float(tolerance),
        "mean_slope_restored": [round(mean_slope_x, 6), round(mean_slope_y, 6)],
    }


def integrate_cumulative_average(slope_x: Floats, slope_y: Floats, valid: Floats) -> Floats:
    """Average the row-wise and column-wise integrals.

    This cancels part of the drift that row alignment accumulates on a curved
    surface, at the price of halving a linear ramp, which the measured table shows.
    """
    along_rows = _along_rows(slope_x, valid)
    along_columns = _along_columns(slope_y, valid)
    height = len(along_rows)
    width = len(along_rows[0])
    return [
        [
            0.5 * (along_rows[r][c] + along_columns[r][c]) if valid[r][c] else 0.0
            for c in range(width)
        ]
        for r in range(height)
    ]


def integrate_height(
    normals: Normals,
    mask: Floats | None = None,
    *,
    method: str = DEFAULT_METHOD,
    iterations: int = DEFAULT_ITERATIONS,
    tolerance: float = DEFAULT_TOLERANCE,
    pixel_scale: float = 1.0,
    min_cosine: float = 0.0,
) -> dict:
    """Height map from a normal field, with the offset that is reported removed.

    A normal field carries orientation, not world size: pixel_scale declares how
    many world units one pixel spans, and the integral is multiplied by it. With
    the default of one, the result is in pixel units.
    The default method is the one with the lowest measured error over the
    controlled geometries; least-squares stays available because it reproduces a
    constant slope exactly.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of: {', '.join(METHODS)}")
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)):
        raise ValueError("tolerance must be numeric")
    if not math.isfinite(float(tolerance)) or float(tolerance) <= 0.0:
        raise ValueError("tolerance must be positive and finite")
    if isinstance(pixel_scale, bool) or not isinstance(pixel_scale, (int, float)):
        raise ValueError("pixel_scale must be numeric")
    if not math.isfinite(float(pixel_scale)) or float(pixel_scale) <= 0.0:
        raise ValueError("pixel_scale must be positive and finite")
    slope_x, slope_y, valid = gradients(normals, mask, min_cosine=min_cosine)
    convergence = None
    if method == "least-squares":
        field, convergence = integrate_least_squares(
            slope_x, slope_y, valid, iterations=iterations, tolerance=float(tolerance)
        )
    elif method == "cumulative-average":
        field = integrate_cumulative_average(slope_x, slope_y, valid)
    elif method == "cumulative-vertical":
        field = integrate_cumulative_vertical(slope_x, slope_y, valid)
    else:
        field = integrate_cumulative(slope_x, slope_y, valid)
    scale = float(pixel_scale)
    scaled = [
        [field[r][c] * scale for c in range(len(field[0]))] for r in range(len(field))
    ]
    samples = [scaled[r][c] for r in range(len(scaled)) for c in range(len(scaled[0]))
               if valid[r][c]]
    offset = sum(samples) / len(samples) if samples else 0.0
    height_map = [
        [round(scaled[r][c] - offset, 6) if valid[r][c] else 0.0
         for c in range(len(scaled[0]))]
        for r in range(len(scaled))
    ]
    return {
        "height": height_map,
        "valid": valid,
        "method": method,
        "pixel_scale": scale,
        "min_cosine": round(float(min_cosine), 6),
        "offset_removed": round(offset, 6),
        "convergence": convergence,
        "integrability": integrability(slope_x, slope_y, valid),
        "pixels": len(samples),
        "note": "heights are relative: an additive constant is not recoverable",
    }


def height_error(recovered: Floats, truth: Floats, valid: Floats) -> dict:
    """RMSE after removing the best offset, which is all a normal field allows."""
    pairs = [
        (recovered[r][c], truth[r][c])
        for r in range(len(truth))
        for c in range(len(truth[0]))
        if valid[r][c]
    ]
    if not pairs:
        return {"pixels": 0, "rmse": None, "max_abs": None, "offset": None}
    differences = [recovered_value - truth_value for recovered_value, truth_value in pairs]
    offset = sum(differences) / len(differences)
    shifted = [difference - offset for difference in differences]
    scale = [truth_value for _, truth_value in pairs]
    span = max(scale) - min(scale)
    rmse = math.sqrt(sum(value * value for value in shifted) / len(shifted))
    return {
        "pixels": len(pairs),
        "offset": round(offset, 6),
        "rmse": round(rmse, 6),
        "max_abs": round(max(abs(value) for value in shifted), 6),
        "relative_rmse": round(rmse / span, 6) if span > 0 else None,
    }


def analytic_height(kind: str, size: int = 32, *, height: float = 0.6,
                    period: float = 12.0) -> tuple[Floats, Floats]:
    """Known heights for the geometries used to score an integration."""
    if isinstance(size, bool) or not isinstance(size, int) or size < 8:
        raise ValueError("size must be an integer of at least eight")
    center = (size - 1) / 2.0
    radius = size * 0.45
    values = [[0.0] * size for _ in range(size)]
    mask = [[0.0] * size for _ in range(size)]
    step = 2.0 * math.pi / period
    for r in range(size):
        for c in range(size):
            x = (c - center) / radius
            y = (r - center) / radius
            if kind == "sphere":
                squared = x * x + y * y
                if squared > 1.0:
                    continue
                values[r][c] = math.sqrt(max(1.0 - squared, 0.0))
            elif kind == "cone":
                length = math.hypot(x, y)
                if length > 1.0:
                    continue
                values[r][c] = 1.0 - length
            elif kind == "waves":
                values[r][c] = height * math.sin(step * c) * math.sin(step * r)
            elif kind == "tilted-plane":
                # Matches the constant normal (-0.35, -0.2, 1) used by the
                # photometric fixtures: z rises with x and y, not the reverse.
                values[r][c] = 0.35 * x + 0.2 * y
            else:
                raise ValueError(f"unknown geometry: {kind}")
            mask[r][c] = 1.0
    return values, mask
