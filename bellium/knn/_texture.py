"""Deterministic texture measures shared by the k-NN and nano-NN tiers.

These helpers are pure measurement functions. They never decide anything on
their own: the k-NN, nano-NN and hybrid specialists turn the measures into
verdicts, and the deterministic baseline stays available for comparison.
"""

from __future__ import annotations

import math
from typing import Any

from bellium.knn._image import Image, Rgb, clamp_rgb, shape

MIN_PERIOD = 2
PERIODICITY_THRESHOLD = 0.5
# A real period needs a trough at half the lag; a slow drift does not have one.
PERIOD_CONTRAST = 0.15
MIN_PROFILE_STD = 0.01
SHIFT_DIP_FLOOR = 0.6
MIN_SHIFT_PERIOD = 4
AXIS_STRENGTH = 0.9
AXIS_EXACT = 0.999
DIAGONAL_STRENGTH = 0.7
AXIS_TOLERANCE_DEG = 12.0
NORMALIZER = 0.25
QUANTIZATION_STEP = 16

SEAM_CHANNELS = ("gap", "seam", "interior", "edge_step", "seam_max", "hot_rows")
LEVEL = 1.0 / 255.0
VISIBLE_STEP = 0.02


def _luma(pixel: tuple[int, int, int]) -> float:
    return (0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.0722 * pixel[2]) / 255.0


def luma_rows(image: Image) -> list[list[float]]:
    height, width = shape(image)
    if height < 4 or width < 4:
        raise ValueError("texture measures need an image of at least 4x4")
    return [[_luma(image[r][c]) for c in range(width)] for r in range(height)]


def column_profile(image: Image) -> list[float]:
    rows = luma_rows(image)
    height = len(rows)
    return [sum(row[c] for row in rows) / height for c in range(len(rows[0]))]


def row_profile(image: Image) -> list[float]:
    rows = luma_rows(image)
    return [sum(row) / len(row) for row in rows]


def autocorrelation(values: list[float]) -> list[float]:
    """Normalized autocorrelation with a small-sample correction per lag."""
    count = len(values)
    if count < 4:
        raise ValueError("autocorrelation needs at least four samples")
    mean = sum(values) / count
    variance = sum((value - mean) ** 2 for value in values)
    if variance <= 0.0:
        # A flat profile has no measurable repeat: do not declare a period.
        return [1.0] + [0.0] * (count // 2)
    result = [1.0]
    for lag in range(1, count // 2 + 1):
        cov = sum((values[i] - mean) * (values[i + lag] - mean) for i in range(count - lag))
        scaled = cov / variance * (count / (count - lag))
        result.append(max(-1.0, min(1.0, scaled)))
    return result


def estimate_period(values: list[float], *, min_period: int = MIN_PERIOD) -> tuple[int | None, float]:
    """Return the smallest strong repeat lag and its strength.

    A peak counts only when it stands clear of the half-lag correlation, so a
    monotonically drifting profile is not mistaken for a repeat, and a profile
    without measurable variation declares no period at all.
    """
    if _profile_std(values) < MIN_PROFILE_STD:
        return None, 0.0
    curve = autocorrelation(values)
    best = 0.0
    for lag in range(min_period, len(curve)):
        value = curve[lag]
        best = max(best, value)
        if value < PERIODICITY_THRESHOLD:
            continue
        left = curve[lag - 1]
        right = curve[lag + 1] if lag + 1 < len(curve) else 0.0
        half = curve[max(lag // 2, 1)]
        if value >= left and value >= right and value - half >= PERIOD_CONTRAST:
            return lag, value
    return None, max(0.0, best)


def _mean_abs_step(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return sum(abs(values[i + 1] - values[i]) for i in range(len(values) - 1)) / (len(values) - 1)


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _axis_seam(rows: list[list[float]]) -> dict[str, float]:
    height = len(rows)
    width = len(rows[0])
    seam_diffs = [abs(rows[r][0] - rows[r][width - 1]) for r in range(height)]
    # Local reference: the largest adjacent step already present at the edges of
    # that row. A sharp but continuous texture must not look broken, and a wrap
    # discontinuity inside high-frequency detail is less visible than its value.
    references = [
        max(abs(rows[r][1] - rows[r][0]), abs(rows[r][width - 1] - rows[r][width - 2]))
        for r in range(height)
    ]
    interior_steps = [
        abs(row[c + 1] - row[c]) for row in rows for c in range(width - 1)
    ]
    seam = sum(seam_diffs) / height
    visible = sum(1 for value in seam_diffs if value > VISIBLE_STEP) / height
    reference = _median(references)
    gap = max(0.0, min(1.0, (seam - reference) / max(reference, LEVEL))) if visible else 0.0
    hot_rows = sum(
        1 for seam_row, reference in zip(seam_diffs, references)
        if seam_row > reference + 0.02
    ) / height
    return {
        "gap": gap,
        "seam": min(seam / NORMALIZER, 1.0),
        "interior": min((sum(interior_steps) / len(interior_steps)) / NORMALIZER, 1.0),
        "edge_step": min(reference / NORMALIZER, 1.0),
        "seam_max": min(max(seam_diffs) / NORMALIZER, 1.0),
        "hot_rows": min(hot_rows, 1.0),
    }


def seam_features(image: Image) -> dict[str, dict[str, float]]:
    """Wrap-seam measures per axis. A high gap means the texture does not meet itself."""
    rows = luma_rows(image)
    transposed = [list(column) for column in zip(*rows)]
    return {"x": _axis_seam(rows), "y": _axis_seam(transposed)}


def deterministic_seam_verdict(features: dict[str, float], *, gap_limit: float = 0.25,
                               seam_limit: float = 0.04) -> str:
    """Documented baseline kept beside the neural tiers for comparison.

    A discontinuity counts only when it exceeds the local edge gradient and is
    large enough to be visible. High-frequency noise therefore wraps cleanly.
    """
    gap = float(features["gap"])
    seam = float(features["seam"])
    return "mismatch" if gap > gap_limit and seam > seam_limit else "continuous"


def _profile_std(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def period_estimates(image: Image) -> dict[str, Any]:
    """Deterministic repeat estimate per axis. The k-NN never replaces it."""
    estimate: dict[str, Any] = {}
    for axis, angle in (("x", 0.0), ("y", 90.0)):
        period, strength = shift_period(image, angle)
        estimate[f"period_{axis}"] = period
        estimate[f"strength_{axis}"] = round(min(max(strength, 0.0), 1.0), 6)
    return estimate


def _sample_bilinear(rows: list[list[float]], r: float, c: float) -> float:
    """Sub-pixel sample, so an oblique shift is not biased by rounding."""
    height = len(rows)
    width = len(rows[0])
    r = min(max(r, 0.0), height - 1.0)
    c = min(max(c, 0.0), width - 1.0)
    r0 = min(int(math.floor(r)), height - 2) if height > 1 else 0
    c0 = min(int(math.floor(c)), width - 2) if width > 1 else 0
    fraction_r = r - r0
    fraction_c = c - c0
    top = rows[r0][c0] * (1.0 - fraction_c) + rows[r0][c0 + 1] * fraction_c
    bottom = (
        rows[r0 + 1][c0] * (1.0 - fraction_c) + rows[r0 + 1][c0 + 1] * fraction_c
    )
    return top * (1.0 - fraction_r) + bottom * fraction_r


def shift_period(image: Image, angle_deg: float, *, max_shift: int | None = None) -> tuple[int | None, float]:
    """Repeat period from the mean absolute difference between shifted copies.

    Unlike a mean projection, this keeps phase information, so a symmetric
    checkerboard is measured correctly. A direction with no variation, or with a
    difference that never dips, declares no period.
    """
    if isinstance(angle_deg, bool) or not isinstance(angle_deg, (int, float)):
        raise ValueError("angle_deg must be numeric")
    if not math.isfinite(float(angle_deg)):
        raise ValueError("angle_deg must be finite")
    rows = luma_rows(image)
    height = len(rows)
    width = len(rows[0])
    limit = max_shift if max_shift is not None else min(height, width) // 2
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= MIN_PERIOD:
        raise ValueError("the shift range is too small to measure a period")
    radians = math.radians(float(angle_deg))
    step_x = math.cos(radians)
    step_y = math.sin(radians)
    differences = [0.0] * (limit + 1)
    for shift in range(1, limit + 1):
        total = 0.0
        count = 0
        for r in range(height):
            shifted_r = r + shift * step_y
            if not 0.0 <= shifted_r <= height - 1.0:
                continue
            row = rows[r]
            for c in range(width):
                shifted_c = c + shift * step_x
                if not 0.0 <= shifted_c <= width - 1.0:
                    continue
                total += abs(row[c] - _sample_bilinear(rows, shifted_r, shifted_c))
                count += 1
        differences[shift] = total / count if count else float("nan")
    usable = [value for value in differences[1:] if math.isfinite(value)]
    if not usable:
        return None, 0.0
    average = sum(usable) / len(usable)
    if average <= MIN_PROFILE_STD:
        # The image does not change along this direction at all.
        return None, 0.0
    for shift in range(MIN_SHIFT_PERIOD, limit + 1):
        value = differences[shift]
        if not math.isfinite(value):
            continue
        right = differences[shift + 1] if shift + 1 <= limit else float("inf")
        if value > differences[shift - 1] or value > right:
            continue
        if value > SHIFT_DIP_FLOOR * average:
            continue
        # The smallest dip wins: a multiple of the true period can dip a hair
        # lower by rounding, and reporting twice the period would be wrong.
        return shift, min(1.0, 1.0 - value / average)
    return None, 0.0


def projection_profile(image: Image, angle_deg: float) -> list[float]:
    """Mean luma binned along the repeat direction, so any angle can be measured.

    Angle 0 repeats along x (vertical stripes) and 90 along y, matching the axis
    profiles. Empty bins at oblique angles are filled by linear interpolation.
    """
    if isinstance(angle_deg, bool) or not isinstance(angle_deg, (int, float)):
        raise ValueError("angle_deg must be numeric")
    if not math.isfinite(float(angle_deg)):
        raise ValueError("angle_deg must be finite")
    rows = luma_rows(image)
    height = len(rows)
    width = len(rows[0])
    radians = math.radians(float(angle_deg))
    cos_a = math.cos(radians)
    sin_a = math.sin(radians)
    coordinates = [c * cos_a + r * sin_a for r in range(height) for c in range(width)]
    low = min(coordinates)
    high = max(coordinates)
    bins = int(math.floor(high - low)) + 1
    if bins < 4:
        raise ValueError("the projection needs at least four bins")
    totals = [0.0] * bins
    counts = [0] * bins
    index = 0
    for r in range(height):
        for c in range(width):
            position = int(coordinates[index] - low)
            position = min(max(position, 0), bins - 1)
            totals[position] += rows[r][c]
            counts[position] += 1
            index += 1
    profile = [
        totals[i] / counts[i] if counts[i] else float("nan") for i in range(bins)
    ]
    return _fill_gaps(profile)


def _fill_gaps(profile: list[float]) -> list[float]:
    known = [i for i, value in enumerate(profile) if not math.isnan(value)]
    if not known:
        raise ValueError("the projection has no covered bin")
    for i, value in enumerate(profile):
        if not math.isnan(value):
            continue
        before = [j for j in known if j < i]
        after = [j for j in known if j > i]
        if not before:
            profile[i] = profile[after[0]]
        elif not after:
            profile[i] = profile[before[-1]]
        else:
            left, right = before[-1], after[0]
            weight = (i - left) / (right - left)
            profile[i] = profile[left] * (1.0 - weight) + profile[right] * weight
    return profile


def axis_of(angle_deg: float) -> str | None:
    """Which image axis a direction belongs to, if any."""
    if angle_deg <= AXIS_TOLERANCE_DEG or angle_deg >= 180.0 - AXIS_TOLERANCE_DEG:
        return "x"
    if abs(angle_deg - 90.0) <= AXIS_TOLERANCE_DEG:
        return "y"
    return None


def direction_scan(
    image: Image,
    angles: tuple[float, ...] | None = None,
    *,
    max_shift: int | None = None,
) -> dict[str, Any]:
    """Repeat period and strength per candidate direction.

    Directions come from the projection method, which keeps phase and therefore
    resolves a striped pattern at any angle. The two image axes are confirmed by
    the shift method, which is the only one that can see a symmetric pattern such
    as a checkerboard. The true repeat direction carries the smallest period: a
    direction tilted by phi only comes back into phase after period / cos(phi).
    """
    candidates = angles or tuple(float(angle) for angle in range(0, 180, 15))
    measured = []
    for angle in candidates:
        period, strength = estimate_period(projection_profile(image, angle))
        measured.append({
            "angle_deg": float(angle),
            "period": period,
            "strength": round(min(max(strength, 0.0), 1.0), 6),
            "method": "profile",
        })
    strong = [
        item for item in measured
        if item["period"] is not None and item["strength"] >= PERIODICITY_THRESHOLD
    ]
    # Only clearly periodic directions compete for "shortest": an oblique
    # projection can show a weak half-period that must not outrank a real axis.
    reliable = [item for item in strong if item["strength"] >= DIAGONAL_STRENGTH]
    smallest = min(reliable, key=lambda item: (item["period"], -item["strength"]), default=None)
    confirmed: dict[str, dict[str, Any] | None] = {}
    for name, angle in (("x", 0.0), ("y", 90.0)):
        period, strength = shift_period(image, angle, max_shift=max_shift)
        entry = {
            "angle_deg": angle,
            "period": period,
            "strength": round(min(max(strength, 0.0), 1.0), 6),
            "method": "shift",
        }
        confirmed[name] = (
            entry if period is not None and entry["strength"] >= AXIS_STRENGTH else None
        )
    tile = (
        confirmed["x"] is not None and confirmed["y"] is not None
        and confirmed["x"]["strength"] >= AXIS_EXACT
        and confirmed["y"]["strength"] >= AXIS_EXACT
    )
    if tile:
        # A checkerboard is periodic on both axes and on the diagonal. Its axis
        # dips are exact, which is what distinguishes a tile from a striped
        # pattern that merely re-phases along an axis.
        axes = {"x": confirmed["x"], "y": confirmed["y"]}
    else:
        axes = {}
        for name in ("x", "y"):
            entry = confirmed[name]
            if entry is None:
                axes[name] = None
                continue
            shortest = smallest is None or entry["period"] <= smallest["period"]
            axes[name] = entry if shortest else None
    diagonal = None if (axes["x"] or axes["y"]) else min(
        (item for item in reliable if axis_of(item["angle_deg"]) is None),
        key=lambda item: (item["period"], -item["strength"]),
        default=None,
    )
    return {
        "angles": measured,
        "axis_x": axes["x"],
        "axis_y": axes["y"],
        "diagonal": diagonal,
        "best": smallest,
    }


def local_period_variation(image: Image, angle_deg: float, *, bands: int = 3) -> dict[str, Any]:
    """Period measured in horizontal strips, to expose perspective or a gradient.

    The shift method is used again here: a mean projection can show a half-period
    on a tiled block and would then claim a variation that is not there.
    """
    if isinstance(bands, bool) or not isinstance(bands, int) or bands < 2:
        raise ValueError("bands must be an integer of at least two")
    rows = luma_rows(image)
    height = len(rows)
    width = len(rows[0])
    if height < bands * 4:
        raise ValueError("the image is too short for this many bands")
    size = height // bands
    radians = math.radians(float(angle_deg))
    horizontal = abs(math.cos(radians)) >= abs(math.sin(radians))
    periods = []
    strengths = []
    for index in range(bands):
        start = index * size
        stop = height if index == bands - 1 else start + size
        strip = _strip_image(rows, start, stop)
        extent = width if horizontal else (stop - start)
        max_shift = max(MIN_SHIFT_PERIOD + 1, extent // 2)
        period, strength = shift_period(strip, angle_deg, max_shift=max_shift)
        periods.append(period)
        strengths.append(round(min(max(strength, 0.0), 1.0), 6))
    known = [period for period in periods if period]
    if len(known) < 2:
        return {
            "periods": periods,
            "strengths": strengths,
            "ratio": None,
            "varies": False,
            "reason": "too_few_measured_bands",
        }
    ratio = max(known) / min(known)
    return {
        "periods": periods,
        "strengths": strengths,
        "ratio": round(ratio, 6),
        "varies": ratio > 1.15,
        "reason": None,
    }


def _strip_image(rows: list[list[float]], start: int, stop: int) -> Image:
    """Rebuild a grey strip so the projection and the image contract both apply."""
    strip: Image = []
    for row in rows[start:stop]:
        line: list[Rgb] = []
        for value in row:
            level = clamp_rgb(value * 255.0)
            line.append((level, level, level))
        strip.append(line)
    return strip


def texture_features(image: Image) -> dict[str, float]:
    """Compact repeat-detection features in [0, 1]. Periods stay separate."""
    rows = luma_rows(image)
    columns = [list(column) for column in zip(*rows)]
    flat = [value for row in rows for value in row]
    gradient = (_mean_abs_step(rows[0]) + _mean_abs_step(columns[0])) / 2.0
    quantized = {(int(value * 255) // QUANTIZATION_STEP) for value in flat}
    estimates = period_estimates(image)
    strength_x = estimates["strength_x"]
    strength_y = estimates["strength_y"]
    return {
        "strength_x": strength_x,
        "strength_y": strength_y,
        "axis_balance": round(max(0.0, 1.0 - abs(strength_x - strength_y)), 6),
        "gradient_energy": round(min(gradient / NORMALIZER, 1.0), 6),
        "profile_variance": round(
            min(max(_profile_std(column_profile(image)), _profile_std(row_profile(image))) / NORMALIZER, 1.0),
            6,
        ),
        "detail_ratio": round(min(len(quantized) / max(len(flat), 1), 1.0), 6),
    }
