"""Bounded bilinear RGB previews with an unchanged context k-NN fallback.

Four known corners define an interpolant. Every other known pixel in the
surrounding rectangle must agree within one stored RGB code value. Agreement
cannot establish the contents of a hidden detail: every fill still needs review.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time

from PIL import Image

from .patch_knn import (
    MAX_CANVAS_PIXELS, MAX_MASKED_PIXELS, InpaintMetrics, InpaintResult,
    inpaint_patch_knn,
)
from .router import InpaintRouteVerdict

INTERPOLATION_METHOD = "bilinear_rgb_v1"
PREVIEW_METHODS = frozenset({"patch_knn", "bilinear_rgb"})
MAX_INTERPOLATION_SPAN = 16
CONTEXT_MARGIN = 2
MAX_RGB_RESIDUAL = 1


@dataclass
class InterpolationGate:
    accepted: bool
    reason: str
    support_pixels: int = 0
    validation_pixels: int = 0
    max_rgb_residual: float | None = None


@dataclass
class PreviewMetrics(InpaintMetrics):
    interpolation: InterpolationGate | None = None


def _validate(image, mask, patch_size, search_radius, k_neighbors,
              max_comparisons, max_context_error):
    # Validate even when interpolation succeeds. Keep the frozen v1 module intact.
    if image.size != mask.size:
        raise ValueError("Image and defect mask dimensions must match")
    if image.mode not in ("RGB", "RGBA"):
        raise ValueError("Inpaint expects RGB or RGBA pixels")
    if not image.width or not image.height or image.width * image.height > MAX_CANVAS_PIXELS:
        raise ValueError("Inpaint accepts 1..262144 image pixels")
    for name, value, limit in (("patch_size", patch_size, 15), ("search_radius", search_radius, 64),
                               ("k_neighbors", k_neighbors, 16), ("max_comparisons", max_comparisons, 2_000_000)):
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= limit:
            raise ValueError(f"{name} must be an integer in 1..{limit}")
    if patch_size < 3 or patch_size % 2 == 0:
        raise ValueError("patch_size must be odd and at least 3")
    if (isinstance(max_context_error, bool) or not isinstance(max_context_error, (float, int))
            or not math.isfinite(max_context_error) or not 0 <= max_context_error <= 1):
        raise ValueError("max_context_error must be finite and within [0, 1]")


def _interpolate(image: Image.Image, selected: Image.Image, count: int):
    def decline(reason):
        return None, InterpolationGate(False, reason)

    if count == 0:
        return decline("empty_selection")
    if count > MAX_MASKED_PIXELS or count / (image.width * image.height) > 0.25:
        return decline("selection_limit")
    left, top, right, bottom = selected.getbbox()
    if right - left > MAX_INTERPOLATION_SPAN or bottom - top > MAX_INTERPOLATION_SPAN:
        return decline("selection_span")
    x0, y0 = left - CONTEXT_MARGIN, top - CONTEXT_MARGIN
    x1, y1 = right - 1 + CONTEXT_MARGIN, bottom - 1 + CONTEXT_MARGIN
    if x0 < 0 or y0 < 0 or x1 >= image.width or y1 >= image.height:
        return decline("missing_enclosing_context")

    src, mask = image.load(), selected.load()
    corners = ((x0, y0), (x1, y0), (x0, y1), (x1, y1))
    if image.mode == "RGBA":
        alpha = src[x0, y0][3]
        if alpha == 0 or any(src[x, y][3] != alpha
                             for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)):
            return decline("nonuniform_or_zero_alpha")
    colors = [src[xy] for xy in corners]
    denominator = (x1 - x0) * (y1 - y0)

    def numerators(x, y):
        weights = ((x1 - x) * (y1 - y), (x - x0) * (y1 - y),
                   (x1 - x) * (y - y0), (x - x0) * (y - y0))
        return tuple(sum(weights[i] * colors[i][c] for i in range(4)) for c in range(3))

    # Integer residuals and rounding keep decisions/output independent of BLAS
    # and floating-point tie behavior. Selected RGB never enters this gate.
    worst = 0
    validated = 0
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            if mask[x, y] or (x, y) in corners:
                continue
            predicted = numerators(x, y)
            worst = max(worst, *(abs(predicted[c] - src[x, y][c] * denominator) for c in range(3)))
            validated += 1
    accepted = worst <= MAX_RGB_RESIDUAL * denominator
    gate = InterpolationGate(accepted, "supported_context" if accepted else "context_mismatch",
                             validated + 4, validated, worst / denominator)
    if not accepted:
        return None, gate

    output = image.copy()
    out = output.load()
    for y in range(top, bottom):
        for x in range(left, right):
            if mask[x, y]:
                # Nonnegative convex combination, nearest integer, ties upwards.
                rgb = tuple((2 * n + denominator) // (2 * denominator) for n in numerators(x, y))
                out[x, y] = rgb if image.mode == "RGB" else (*rgb, src[x, y][3])
    return output, gate


def inpaint_preview(
    image: Image.Image,
    mask: Image.Image,
    *,
    patch_size: int = 5,
    search_radius: int = 25,
    k_neighbors: int = 3,
    max_comparisons: int = 200_000,
    max_context_error: float = 0.05,
) -> InpaintResult:
    """Try a strictly checked interpolant, otherwise run context k-NN v1.

    Interpolation has fixed limits: selection bbox <=16x16, two known pixels
    of margin on every side, uniform positive local alpha, maximum RGB residual
    <=1 across all known context. At most a 20x20 rectangle is examined.
    Keyword budgets apply to the fallback only and are always validated.
    The selected method and interpolation gate are included in metrics.
    """
    started = time.perf_counter()
    _validate(image, mask, patch_size, search_radius, k_neighbors, max_comparisons, max_context_error)
    selected = mask.convert("L").point(lambda value: 255 if value > 128 else 0)
    count = sum(selected.histogram()[129:])
    output, gate = _interpolate(image, selected, count)
    if output is not None:
        ratio = count / (image.width * image.height)
        verdict = InpaintRouteVerdict("bilinear_rgb", 0.0, ratio,
                                      "Known context supports bilinear RGB interpolation; review required")
        metrics = PreviewMetrics(count, ratio, verdict, 0, method=INTERPOLATION_METHOD,
                                 interpolation=gate)
        result = InpaintResult(output, metrics)
    else:
        result = inpaint_patch_knn(image, selected, patch_size=patch_size, search_radius=search_radius,
                                   k_neighbors=k_neighbors, max_comparisons=max_comparisons,
                                   max_context_error=max_context_error)
        result.metrics = PreviewMetrics(**vars(result.metrics), interpolation=gate)
    result.metrics.elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
    return result
