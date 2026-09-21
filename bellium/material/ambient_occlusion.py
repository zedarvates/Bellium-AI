"""Horizon-based ambient occlusion from a relative height field.

Ambient occlusion measures the fraction of the hemisphere that is occluded by the
surrounding terrain or geometry. For a height field H(r, c), rays are traced outwards
along discrete horizon directions up to a declared sampling radius. The horizon
elevation angle along each direction gives the cosine-weighted occlusion.

The output is relative and depends on the declared pixel scale, exactly like surface
height: orientation and relative elevation are recovered, but world units must be
declared by the caller.
"""

from __future__ import annotations

import math
from typing import Any

Floats = list[list[float]]

DEFAULT_RADIUS = 6
DEFAULT_DIRECTIONS = 8
MIN_RADIUS = 1
MAX_RADIUS = 64
MIN_DIRECTIONS = 4
MAX_DIRECTIONS = 32


def _validate_inputs(
    height: Floats,
    valid: Floats | None,
    radius: int,
    directions: int,
) -> tuple[int, int, int, int]:
    if not isinstance(height, list) or not height or not height[0]:
        raise ValueError("height must be a non-empty grid")
    rows = len(height)
    cols = len(height[0])
    for r, row in enumerate(height):
        if len(row) != cols:
            raise ValueError("height rows must have equal width")
        for c, val in enumerate(row):
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise ValueError(f"height[{r}][{c}] must be numeric")
            if not math.isfinite(float(val)):
                raise ValueError(f"height[{r}][{c}] must be finite")
    if valid is not None:
        if len(valid) != rows or any(len(row) != cols for row in valid):
            raise ValueError("valid mask must share the height shape")
    if isinstance(radius, bool) or not isinstance(radius, int) or radius < MIN_RADIUS or radius > MAX_RADIUS:
        raise ValueError(f"radius must be an integer between {MIN_RADIUS} and {MAX_RADIUS}")
    if isinstance(directions, bool) or not isinstance(directions, int) or directions < MIN_DIRECTIONS or directions > MAX_DIRECTIONS:
        raise ValueError(f"directions must be an integer between {MIN_DIRECTIONS} and {MAX_DIRECTIONS}")
    return rows, cols, radius, directions


def horizon_ambient_occlusion(
    height: Floats,
    valid: Floats | None = None,
    *,
    radius: int = DEFAULT_RADIUS,
    directions: int = DEFAULT_DIRECTIONS,
    pixel_scale: float = 1.0,
    elevation_scale: float = 1.0,
) -> dict[str, Any]:
    """Calculate horizon-based ambient occlusion from a height field.

    Returns an accessibility factor in [0.0, 1.0], where 1.0 is completely open sky
    (no occlusion) and 0.0 is fully occluded.

    pixel_scale: world-unit width per pixel.
    elevation_scale: multiplier applied to height values to align with pixel units.
    """
    rows, cols, rad, num_dirs = _validate_inputs(height, valid, radius, directions)
    if isinstance(pixel_scale, bool) or not isinstance(pixel_scale, (int, float)) or float(pixel_scale) <= 0.0 or not math.isfinite(float(pixel_scale)):
        raise ValueError("pixel_scale must be positive and finite")
    if isinstance(elevation_scale, bool) or not isinstance(elevation_scale, (int, float)) or float(elevation_scale) <= 0.0 or not math.isfinite(float(elevation_scale)):
        raise ValueError("elevation_scale must be positive and finite")

    scale_ratio = float(elevation_scale) / float(pixel_scale)

    mask = (
        [[1.0] * cols for _ in range(rows)]
        if valid is None
        else [[1.0 if valid[r][c] > 0.0 else 0.0 for c in range(cols)] for r in range(rows)]
    )

    ao: Floats = [[1.0] * cols for _ in range(rows)]
    samples_count = 0
    total_occlusion_sum = 0.0

    # Precalculate direction unit vectors
    dir_vectors = [
        (math.cos(2.0 * math.pi * d / num_dirs), math.sin(2.0 * math.pi * d / num_dirs))
        for d in range(num_dirs)
    ]

    for r in range(rows):
        for c in range(cols):
            if mask[r][c] <= 0.0:
                ao[r][c] = 0.0
                continue

            h0 = float(height[r][c])
            total_occ = 0.0

            for dx, dy in dir_vectors:
                max_tan = 0.0
                for step in range(1, rad + 1):
                    rr = int(round(r + dy * step))
                    cc = int(round(c + dx * step))
                    if 0 <= rr < rows and 0 <= cc < cols and mask[rr][cc] > 0.0:
                        dh = (float(height[rr][cc]) - h0) * scale_ratio
                        if dh > 0.0:
                            tan_val = dh / float(step)
                            if tan_val > max_tan:
                                max_tan = tan_val

                # Horizon elevation angle psi = atan(max_tan)
                # Projected solid-angle occlusion = sin(psi)
                psi = math.atan(max_tan)
                total_occ += math.sin(psi)

            mean_occ = total_occ / float(num_dirs)
            accessibility = max(0.0, min(1.0, 1.0 - mean_occ))
            ao[r][c] = round(accessibility, 6)
            total_occlusion_sum += mean_occ
            samples_count += 1

    mean_ao = (
        round(1.0 - (total_occlusion_sum / samples_count), 6)
        if samples_count > 0
        else 0.0
    )

    return {
        "ao": ao,
        "valid": mask,
        "radius": rad,
        "directions": num_dirs,
        "pixel_scale": float(pixel_scale),
        "elevation_scale": float(elevation_scale),
        "mean_ao": mean_ao,
        "pixels": samples_count,
        "units": "accessibility [0.0 fully occluded, 1.0 fully open]",
    }

