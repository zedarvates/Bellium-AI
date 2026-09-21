"""Deterministic X/Y texture repetition analysis.

The scan compares the image with shifts of itself along each axis (mean
absolute difference, so a match score of 1.0 means identical). A period is
reported only when the best score is high and is either a near-perfect peak
consistent with its double (harmonic check) or clearly above its neighbouring
shifts without sitting on the scan boundary. Otherwise the axis abstains.
Analysis is bounded by downscaling the longest side to "max_analysis", and
reported periods are converted back to source pixels.

This is a deterministic measurement with no learned weights. It reports
periods in pixels and wrapped-edge difference (0 = identical edges, 1 =
maximally different); it does not synthesize a seamless tile and it is not
ground truth about how a texture was authored.
"""
from __future__ import annotations

from dataclasses import dataclass
import statistics

from PIL import Image, ImageChops, ImageStat

MIN_ANALYSIS = 16


class TextureError(ValueError):
    """Raised when a texture analysis request is invalid."""


@dataclass(frozen=True)
class AxisRepeat:
    axis: str
    period_px: int | None
    period_analysis: int | None
    confidence: float
    match: float
    margin: float
    median_match: float
    reason: str
    top_periods: tuple[tuple[int, float], ...]

    def to_dict(self) -> dict:
        return {
            "axis": self.axis,
            "period_px": self.period_px,
            "period_analysis": self.period_analysis,
            "confidence": self.confidence,
            "match": self.match,
            "margin": self.margin,
            "median_match": self.median_match,
            "reason": self.reason,
            "top_periods": [[period, score] for period, score in self.top_periods],
        }


@dataclass(frozen=True)
class RepeatReport:
    requested_axis: str
    x: AxisRepeat
    y: AxisRepeat
    analysis_size: tuple[int, int]
    scale_x: float
    scale_y: float
    edge_difference_x: float
    edge_difference_y: float
    reliable: bool

    def to_dict(self) -> dict:
        return {
            "requested_axis": self.requested_axis,
            "reliable": self.reliable,
            "analysis_size": list(self.analysis_size),
            "scale_x": self.scale_x,
            "scale_y": self.scale_y,
            "edge_difference_x": self.edge_difference_x,
            "edge_difference_y": self.edge_difference_y,
            "x": self.x.to_dict(),
            "y": self.y.to_dict(),
        }


def _validate_axis(axis: str) -> str:
    if axis not in ("both", "x", "y"):
        raise TextureError("axis must be 'both', 'x' or 'y'.")
    return axis


def _validate_int(value, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TextureError(f"{name} must be an integer greater than or equal to {minimum}.")
    return value


def _validate_ratio(value, name: str, upper_exclusive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TextureError(f"{name} must be a number between 0 and 1.")
    value = float(value)
    upper_ok = value < 1.0 if upper_exclusive else value <= 1.0
    if not 0.0 <= value <= 1.0 or not upper_ok:
        raise TextureError(f"{name} must be a number between 0 and 1.")
    return value


def _match_score(gray: Image.Image, axis: str, period: int) -> float:
    width, height = gray.size
    if axis == "x":
        left = gray.crop((0, 0, width - period, height))
        right = gray.crop((period, 0, width, height))
    else:
        left = gray.crop((0, 0, width, height - period))
        right = gray.crop((0, period, width, height))
    mean = ImageStat.Stat(ImageChops.difference(left, right)).mean[0]
    return 1.0 - mean / 255.0


def _edge_difference(gray: Image.Image, axis: str) -> float:
    width, height = gray.size
    if axis == "x":
        if width < 2:
            return 0.0
        first = gray.crop((0, 0, 1, height))
        last = gray.crop((width - 1, 0, width, height))
    else:
        if height < 2:
            return 0.0
        first = gray.crop((0, 0, width, 1))
        last = gray.crop((0, height - 1, width, height))
    return ImageStat.Stat(ImageChops.difference(first, last)).mean[0] / 255.0


def _axis_report(
    gray: Image.Image,
    axis: str,
    min_period: int,
    max_period: int | None,
    min_match: float,
    min_contrast: float,
    scale: float,
) -> AxisRepeat:
    size = gray.size[0] if axis == "x" else gray.size[1]
    limit = max_period if max_period is not None else max(min_period, size // 2)
    limit = min(limit, size - 1)
    scores = {
        period: _match_score(gray, axis, period)
        for period in range(min_period, limit + 1)
    }
    if not scores:
        return AxisRepeat(axis, None, None, 0.0, 0.0, 0.0, 0.0, "no_candidate_period", ())
    median = statistics.median(scores.values())
    best_period = max(scores, key=lambda period: (scores[period], -period))
    best = scores[best_period]
    window = max(2, best_period // 4)
    neighbours = [
        score
        for period, score in scores.items()
        if period != best_period and abs(period - best_period) <= window
    ]
    margin = best - (max(neighbours) if neighbours else median)
    top = tuple(sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:3])
    half = best_period // 2
    anti_phase_ok = half < min_period or best - scores[half] >= min_contrast
    variation_ok = min(scores.values()) <= 0.95
    doubled = best_period * 2
    harmonic_ok = doubled not in scores or scores[doubled] >= 0.95
    near_perfect = best >= 0.99 and harmonic_ok
    on_boundary = best_period == min_period or best_period == limit
    boundary_ok = (not on_boundary) or best >= 0.99
    accepted = (
        best >= min_match
        and variation_ok
        and anti_phase_ok
        and (near_perfect or (margin >= min_contrast and boundary_ok))
    )
    if accepted:
        match_part = (best - min_match) / max(1e-9, 1.0 - min_match)
        contrast_part = min(1.0, margin / 0.10)
        confidence = min(1.0, 0.35 * match_part + 0.65 * contrast_part)
        if near_perfect:
            confidence = max(confidence, 0.9)
        confidence = round(confidence, 3)
        period_px = max(1, int(round(best_period / scale)))
        return AxisRepeat(
            axis, period_px, best_period, confidence,
            round(best, 4), round(margin, 4), round(median, 4), "reliable_period", top,
        )
    return AxisRepeat(
        axis, None, best_period, 0.0,
        round(best, 4), round(margin, 4), round(median, 4), "no_reliable_period", top,
    )


def detect_repeat(
    image: Image.Image,
    *,
    axis: str = "both",
    min_period: int = 2,
    max_period: int | None = None,
    min_match: float = 0.8,
    min_contrast: float = 0.05,
    max_analysis: int = 256,
) -> RepeatReport:
    """Measure X/Y repetition with confidence, margins and abstention."""
    if not isinstance(image, Image.Image):
        raise TextureError("image must be a PIL.Image.Image instance.")
    axis = _validate_axis(axis)
    min_period = _validate_int(min_period, "min_period", 2)
    if max_period is not None:
        max_period = _validate_int(max_period, "max_period", min_period)
    min_match = _validate_ratio(min_match, "min_match", upper_exclusive=True)
    min_contrast = _validate_ratio(min_contrast, "min_contrast")
    max_analysis = _validate_int(max_analysis, "max_analysis", MIN_ANALYSIS)

    source = image.convert("L")
    width, height = source.size
    if width < 4 or height < 4:
        raise TextureError("image must be at least 4x4 pixels.")

    analysis = source
    scale_x = 1.0
    scale_y = 1.0
    if max(width, height) > max_analysis:
        ratio = max_analysis / max(width, height)
        new_size = (max(4, int(round(width * ratio))), max(4, int(round(height * ratio))))
        analysis = source.resize(new_size, Image.Resampling.BILINEAR)
        scale_x = new_size[0] / width
        scale_y = new_size[1] / height

    x_result = _axis_report(analysis, "x", min_period, max_period, min_match, min_contrast, scale_x)
    y_result = _axis_report(analysis, "y", min_period, max_period, min_match, min_contrast, scale_y)
    reliable = False
    if axis in ("x", "both") and x_result.period_px is not None:
        reliable = True
    if axis in ("y", "both") and y_result.period_px is not None:
        reliable = True

    return RepeatReport(
        requested_axis=axis,
        x=x_result,
        y=y_result,
        analysis_size=analysis.size,
        scale_x=round(scale_x, 6),
        scale_y=round(scale_y, 6),
        edge_difference_x=round(_edge_difference(analysis, "x"), 4),
        edge_difference_y=round(_edge_difference(analysis, "y"), 4),
        reliable=reliable,
    )
