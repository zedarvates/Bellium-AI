"""Deterministic motion-curve measures for animation timing.

Pure measurement helpers over a series of frame-to-frame change ratios. They
never decide anything on their own: the k-NN, nano and hybrid tiers turn the
measures into verdicts, and the published threshold rules stay beside them.
"""

from __future__ import annotations

import math

SAMPLES = 7
HOLD_THRESHOLD = 0.02
HOLD_RELATIVE = 0.05
HOLD_FLOOR = 0.002
DUPLICATE_MISMATCH = 0.02


def motion_series(values: object, *, minimum: int = 3) -> list[float]:
    """Validated frame-to-frame change ratios."""
    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 1:
        raise ValueError("minimum must be a positive integer")
    if not isinstance(values, (list, tuple)) or len(values) < minimum:
        raise ValueError(f"a motion series needs at least {minimum} transitions")
    series = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"transition {index} must be numeric")
        number = float(value)
        if not math.isfinite(number) or number < 0.0 or number > 1.0:
            raise ValueError(f"transition {index} must be between 0 and 1")
        series.append(number)
    return series


def cumulative(values: list[float]) -> list[float]:
    """Running total, so a translating sprite becomes a progress curve."""
    total = 0.0
    out = []
    for value in values:
        total += value
        out.append(total)
    return out


def progress_samples(values: list[float], samples: int = SAMPLES) -> list[float]:
    """Normalised progress resampled to a fixed number of points in [0, 1].

    The curve starts at the origin of the clip, so the first sample is zero and
    the reference curves can be compared directly.
    """
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 3:
        raise ValueError("samples must be an integer of at least three")
    series = motion_series(values)
    totals = [0.0] + cumulative(series)
    span = totals[-1]
    if span <= 0.0:
        return [0.0] * samples
    last = len(totals) - 1
    out = []
    for index in range(samples):
        target = (index / (samples - 1)) * last
        low = int(math.floor(target))
        high = min(low + 1, last)
        fraction = target - low
        value = totals[low] * (1.0 - fraction) + totals[high] * fraction
        out.append(min(max(value / span, 0.0), 1.5))
    return out


def hold_runs(values: list[float], *, threshold: float | None = None) -> list[dict]:
    """Runs of transitions that count as a hold, as (start, length) pairs.

    The threshold is relative to the strongest transition of the same clip,
    because a small sprite changes far fewer pixels than a large one; a fixed
    absolute threshold would call every frame of a small clip a hold.
    """
    series = motion_series(values)
    if threshold is None:
        threshold = max(HOLD_FLOOR, HOLD_RELATIVE * max(series))
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError("threshold must be numeric")
    if not math.isfinite(float(threshold)) or float(threshold) < 0.0:
        raise ValueError("threshold must be positive and finite")
    runs: list[dict] = []
    start: int | None = None
    for index, value in enumerate(series):
        if value < threshold:
            if start is None:
                start = index
        elif start is not None:
            runs.append({"start": start, "length": index - start})
            start = None
    if start is not None:
        runs.append({"start": start, "length": len(series) - start})
    return runs


def peak_index(values: list[float]) -> int:
    series = motion_series(values)
    return max(range(len(series)), key=series.__getitem__)


def spacing_variation(values: list[float]) -> float:
    """Coefficient of variation of the transitions, as a 0..1 roughness measure."""
    series = motion_series(values)
    mean = sum(series) / len(series)
    if mean <= 0.0:
        return 0.0
    variance = sum((value - mean) ** 2 for value in series) / len(series)
    return min(math.sqrt(variance) / mean, 1.0)


def is_duplicate(mismatch: float, *, limit: float = DUPLICATE_MISMATCH) -> bool:
    if isinstance(mismatch, bool) or not isinstance(mismatch, (int, float)):
        raise ValueError("mismatch must be numeric")
    if not math.isfinite(float(mismatch)) or not 0.0 <= float(mismatch) <= 1.0:
        raise ValueError("mismatch must be between 0 and 1")
    return float(mismatch) <= limit


REFERENCE_CURVES: dict[str, list[float]] = {
    "linear": [0.0, 1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6, 1.0],
    "ease-in": [0.0, 0.028, 0.111, 0.25, 0.444, 0.694, 1.0],
    "ease-out": [0.0, 0.306, 0.556, 0.75, 0.889, 0.972, 1.0],
    "ease-in-out": [0.0, 0.074, 0.259, 0.5, 0.741, 0.926, 1.0],
    "overshoot": [0.0, 0.241, 0.63, 0.95, 1.06, 1.02, 1.0],
}


def curve_error(samples: list[float], curve: list[float]) -> float:
    if len(samples) != len(curve):
        raise ValueError("the sample and curve lengths must match")
    total = sum((sample - reference) ** 2 for sample, reference in zip(samples, curve))
    return math.sqrt(total / len(samples))


def deterministic_easing_verdict(samples: list[float]) -> str:
    """Published baseline: midpoint and overshoot thresholds on the progress curve."""
    if not isinstance(samples, list) or len(samples) < 3:
        raise ValueError("samples must be a list of at least three values")
    middle = samples[len(samples) // 2]
    if max(samples) > 1.02:
        return "overshoot"
    errors = {name: curve_error(samples, curve) for name, curve in REFERENCE_CURVES.items()}
    if min(errors.values()) <= 0.05:
        return min(errors, key=errors.get)
    if middle < 0.35:
        return "ease-in"
    if middle > 0.65:
        return "ease-out"
    return "ease-in-out"
