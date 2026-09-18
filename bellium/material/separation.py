"""Log-domain albedo and illumination separation, scored against known renders.

The illumination level is not identifiable from a single image, so the result is
anchored on a documented percentile and every error is reported twice: absolute
and after the best global scale. Shading with high-frequency content, and a flat
albedo, are the known failure modes and are reported as such.
"""

from __future__ import annotations

import math

from bellium.knn._texture import estimate_period
from bellium.material.controlled import (
    EPSILON,
    ColorFloats,
    Floats,
    to_gray,
)

DEFAULT_RADIUS = 6
DEFAULT_ANCHOR = 0.98
FLAT_VARIANCE = 1e-4
CLIPPED_LIMIT = 0.02


def _validate(render: ColorFloats) -> tuple[int, int]:
    if not isinstance(render, list) or not render or not render[0]:
        raise ValueError("render is empty")
    width = len(render[0])
    if any(len(row) != width for row in render):
        raise ValueError("render rows must have equal width")
    for row in render:
        for pixel in row:
            if len(pixel) != 3:
                raise ValueError("render pixels need three channels")
            for channel in pixel:
                if isinstance(channel, bool) or not isinstance(channel, (int, float)):
                    raise ValueError("render channels must be numeric")
                if not math.isfinite(float(channel)) or not 0.0 <= float(channel) <= 1.0:
                    raise ValueError("render channels must be between 0 and 1")
    return len(render), width


def box_blur(values: Floats, radius: int) -> Floats:
    """Separable box blur with edge clamping, used as the illumination estimate."""
    if isinstance(radius, bool) or not isinstance(radius, int) or radius < 1:
        raise ValueError("radius must be an integer of at least one")
    height = len(values)
    width = len(values[0])
    horizontal = [[0.0] * width for _ in range(height)]
    for r in range(height):
        for c in range(width):
            total = 0.0
            for offset in range(-radius, radius + 1):
                index = min(max(c + offset, 0), width - 1)
                total += values[r][index]
            horizontal[r][c] = total / (2 * radius + 1)
    out = [[0.0] * width for _ in range(height)]
    for c in range(width):
        for r in range(height):
            total = 0.0
            for offset in range(-radius, radius + 1):
                index = min(max(r + offset, 0), height - 1)
                total += horizontal[index][c]
            out[r][c] = total / (2 * radius + 1)
    return out


def _channel(render: ColorFloats, index: int) -> Floats:
    return [[pixel[index] for pixel in row] for row in render]


def _percentile(values: Floats, fraction: float) -> float:
    flat = sorted(value for row in values for value in row)
    if not flat:
        raise ValueError("no values to rank")
    position = min(max(int(round(fraction * (len(flat) - 1))), 0), len(flat) - 1)
    return flat[position]


def separate_albedo(
    render: ColorFloats,
    *,
    radius: int = DEFAULT_RADIUS,
    anchor: float = DEFAULT_ANCHOR,
) -> dict:
    """Split a render into albedo and illumination in linear [0, 1] floats."""
    height, width = _validate(render)
    if isinstance(anchor, bool) or not isinstance(anchor, (int, float)):
        raise ValueError("anchor must be numeric")
    if not 0.5 < float(anchor) <= 1.0:
        raise ValueError("anchor must be in (0.5, 1]")
    albedo: ColorFloats = [[(0.0, 0.0, 0.0) for _ in range(width)] for _ in range(height)]
    illumination: Floats = [[0.0] * width for _ in range(height)]
    channel_anchors = []
    for index in range(3):
        channel = _channel(render, index)
        low = box_blur([[math.log(max(value, EPSILON)) for value in row] for row in channel], radius)
        detail = [
            [math.log(max(channel[r][c], EPSILON)) - low[r][c] for c in range(width)]
            for r in range(height)
        ]
        peak = _percentile(detail, float(anchor))
        channel_anchors.append(round(peak, 6))
        for r in range(height):
            for c in range(width):
                value = min(max(math.exp(detail[r][c] - peak), 0.0), 1.0)
                pixel = list(albedo[r][c])
                pixel[index] = value
                albedo[r][c] = (pixel[0], pixel[1], pixel[2])
                if index == 1:
                    illumination[r][c] = min(max(math.exp(low[r][c] + peak), 0.0), 1.0)
    return {
        "albedo": albedo,
        "illumination": illumination,
        "method": "log-homomorphic-separation",
        "radius": radius,
        "anchor_percentile": float(anchor),
        "channel_anchors": channel_anchors,
    }


def patch_features(
    render: ColorFloats,
    albedo: ColorFloats,
    illumination: Floats,
    *,
    patch: int = 8,
) -> list[dict]:
    """Per-patch statistics behind the trust decision, all in [0, 1]."""
    if isinstance(patch, bool) or not isinstance(patch, int) or patch < 2:
        raise ValueError("patch must be an integer of at least two")
    height, width = _validate(render)
    patches = []
    for row_start in range(0, height - patch + 1, patch):
        for column_start in range(0, width - patch + 1, patch):
            render_values = []
            albedo_values = []
            light_values = []
            clipped = 0
            for r in range(row_start, row_start + patch):
                for c in range(column_start, column_start + patch):
                    light = sum(render[r][c]) / 3.0
                    albedo_value = sum(albedo[r][c]) / 3.0
                    render_values.append(light)
                    albedo_values.append(albedo_value)
                    light_values.append(illumination[r][c])
                    if light >= 0.999:
                        clipped += 1
            count = len(render_values)
            mean = sum(render_values) / count
            variance = sum((value - mean) ** 2 for value in render_values) / count
            albedo_mean = sum(albedo_values) / count
            albedo_variance = (
                sum((value - albedo_mean) ** 2 for value in albedo_values) / count
            )
            gradient = 0.0
            for r in range(row_start, row_start + patch - 1):
                for c in range(column_start, column_start + patch - 1):
                    first = sum(render[r][c]) / 3.0
                    gradient += abs(sum(render[r][c + 1]) / 3.0 - first)
                    gradient += abs(sum(render[r + 1][c]) / 3.0 - first)
            light_min = min(light_values)
            light_max = max(light_values)
            patches.append({
                "row": row_start,
                "column": column_start,
                "features": {
                    "mean_luma": round(min(mean, 1.0), 6),
                    "luma_std": round(min(math.sqrt(variance) * 2.0, 1.0), 6),
                    "gradient_energy": round(min(gradient / (2 * count), 1.0), 6),
                    "albedo_std": round(min(math.sqrt(albedo_variance) * 2.0, 1.0), 6),
                    "illumination_span": round(min(light_max - light_min, 1.0), 6),
                    "clipped_ratio": round(clipped / count, 6),
                },
            })
    return patches


def _rmse(left: ColorFloats, right: ColorFloats) -> float:
    total = 0.0
    count = 0
    for row_left, row_right in zip(left, right):
        for pixel_left, pixel_right in zip(row_left, row_right):
            for index in range(3):
                total += (pixel_left[index] - pixel_right[index]) ** 2
                count += 1
    return math.sqrt(total / count)


def _scale_free(left: ColorFloats, right: ColorFloats) -> tuple[float, float]:
    """Best global scale and the RMSE that remains after applying it."""
    numerator = 0.0
    denominator = 0.0
    for row_left, row_right in zip(left, right):
        for pixel_left, pixel_right in zip(row_left, row_right):
            for index in range(3):
                numerator += pixel_left[index] * pixel_right[index]
                denominator += pixel_right[index] * pixel_right[index]
    scale = numerator / denominator if denominator > 0.0 else 0.0
    scaled = [
        [
            (
                min(max(pixel[0] * scale, 0.0), 1.0),
                min(max(pixel[1] * scale, 0.0), 1.0),
                min(max(pixel[2] * scale, 0.0), 1.0),
            )
            for pixel in row
        ]
        for row in left
    ]
    return round(scale, 6), _rmse(scaled, right)


def separation_error(recovered: dict, truth: ColorFloats) -> dict:
    """Absolute and scale-free albedo error, plus the flat-albedo ratio."""
    scale, scaled_rmse = _scale_free(recovered["albedo"], truth)
    flat_patches = sum(
        1
        for patch in patch_features(
            recovered["albedo"], recovered["albedo"], recovered["illumination"]
        )
        if patch["features"]["albedo_std"] <= math.sqrt(FLAT_VARIANCE) * 2.0
    )
    return {
        "rmse": round(_rmse(recovered["albedo"], truth), 6),
        "best_scale": scale,
        "scale_free_rmse": round(scaled_rmse, 6),
        "flat_patches": flat_patches,
    }


def unaffected_baseline(render: ColorFloats, truth: ColorFloats) -> dict:
    """Score of the naive answer, kept for comparison in every report."""
    scale, scaled_rmse = _scale_free(render, truth)
    return {
        "rmse": round(_rmse(render, truth), 6),
        "best_scale": scale,
        "scale_free_rmse": round(scaled_rmse, 6),
    }


def clipped_ratio(render: ColorFloats) -> float:
    total = 0
    count = 0
    for row in render:
        for pixel in row:
            count += 1
            if sum(pixel) / 3.0 >= 0.999:
                total += 1
    return total / count if count else 0.0


def detail_energy(render: ColorFloats, *, radius: int = DEFAULT_RADIUS) -> float:
    """Relative high-frequency energy of a render, in [0, 1].

    A frequency separation can only move smooth shading: when the image carries
    almost no high-frequency content, the blur removes the albedo itself and the
    identity answer is the honest one.
    """
    luma = to_gray(render)
    smooth = box_blur(luma, radius)
    detail = 0.0
    level = 0.0
    for r, row in enumerate(luma):
        for c, value in enumerate(row):
            detail += abs(value - smooth[r][c])
            level += value
    if level <= 0.0:
        return 0.0
    return min(detail / level, 1.0)


def shading_variation(separated: dict) -> float:
    """Relative spread of the estimated illumination, in [0, 1]."""
    values = [value for row in separated["illumination"] for value in row]
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    if mean <= 0.0:
        return 0.0
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return min(math.sqrt(variance) / mean, 1.0)


def select_method(
    render: ColorFloats,
    *,
    detail_threshold: float = 0.09,
    radius: int = DEFAULT_RADIUS,
) -> str:
    """Documented rule for choosing between separation and the identity answer.

    The threshold sits between a smooth field (about 0.05 at these radii) and a
    structured albedo (about 0.5), so a flat or gradient albedo is not separated.
    Two conditions must hold: the render must carry detail worth keeping, and the
    low-frequency variation must not repeat. Real shading does not repeat; an
    albedo structure does, and blurring it would take the albedo for shading.
    """
    if isinstance(detail_threshold, bool) or not isinstance(detail_threshold, (int, float)):
        raise ValueError("detail_threshold must be numeric")
    if not 0.0 <= float(detail_threshold) < 1.0:
        raise ValueError("detail_threshold must be in [0, 1)")
    if detail_energy(render, radius=radius) < float(detail_threshold):
        return "identity"
    if low_frequency_repeats(render, radius=radius):
        return "identity"
    return "frequency-separation"


def low_frequency_repeats(render: ColorFloats, *, radius: int = DEFAULT_RADIUS) -> bool:
    """True when the blurred image repeats, which means the blur sees albedo.

    The blur used for this test is deliberately smaller than the separation blur:
    a wider one would average a fine albedo pattern away and hide the repetition.
    """
    luma = box_blur(to_gray(render), max(2, radius // 2))
    height = len(luma)
    width = len(luma[0])
    columns = [sum(row[c] for row in luma) / height for c in range(width)]
    rows = [sum(row) / width for row in luma]
    for profile in (columns, rows):
        period, strength = estimate_period(profile)
        if period is not None and strength >= 0.5:
            return True
    return False


def identity_separation(render: ColorFloats) -> dict:
    """The honest fallback: the render is the albedo and the shading is unknown."""
    height, width = _validate(render)
    return {
        "albedo": [[tuple(pixel) for pixel in row] for row in render],
        "illumination": [[1.0] * width for _ in range(height)],
        "method": "identity-no-separation",
        "radius": 0,
        "anchor_percentile": 1.0,
        "channel_anchors": [0.0, 0.0, 0.0],
    }


SHADING_PRIORS = ("smooth", "none", "unknown")
CLIPPED_ABSTENTION = 0.2


def separate_render(
    render: ColorFloats,
    *,
    expected_shading: str = "unknown",
    radius: int = DEFAULT_RADIUS,
) -> dict:
    """Return both candidates, the evidence, and the choice the caller's prior implies.

    A single image cannot say whether its low-frequency variation is shading or
    albedo: a fine albedo pattern and a smooth gradient look the same to a blur.
    With a declared prior the answer is determined; without one the specialist
    reports the evidence and flags the choice as unresolved.
    """
    if expected_shading not in SHADING_PRIORS:
        raise ValueError(f"expected_shading must be one of: {', '.join(SHADING_PRIORS)}")
    separated = separate_albedo(render, radius=radius)
    identity = identity_separation(render)
    evidence = {
        "detail_energy": round(detail_energy(render, radius=radius), 6),
        "low_frequency_repeats": low_frequency_repeats(render, radius=radius),
        "shading_variation": round(shading_variation(separated), 6),
        "clipped_ratio": round(clipped_ratio(render), 6),
    }
    recommended = select_method(render, radius=radius)
    if expected_shading == "smooth":
        chosen, choice = separated, "prior:smooth"
    elif expected_shading == "none":
        chosen, choice = identity, "prior:none"
    else:
        chosen = separated if recommended == "frequency-separation" else identity
        choice = "recommended"
    return {
        "albedo": chosen["albedo"],
        "illumination": chosen["illumination"],
        "method": chosen["method"],
        "choice": choice,
        "recommendation": recommended,
        "unresolved": expected_shading == "unknown",
        "candidates": {
            "identity": {"albedo": identity["albedo"], "illumination": identity["illumination"]},
            "frequency-separation": {
                "albedo": separated["albedo"],
                "illumination": separated["illumination"],
            },
        },
        "evidence": evidence,
        "radius": radius,
        "channel_anchors": separated["channel_anchors"],
    }
