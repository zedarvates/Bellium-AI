"""Bounded context-scored patch k-NN previews using original donor patches.

Only masked RGB values may change; alpha and all other pixels are preserved.
Exhausted support/work budgets roll back the whole preview and ask for review.
Scores and mask routing remain heuristics, never evidence of semantic repair.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
import time

from PIL import Image

from .router import route_inpaint_request, InpaintRouteVerdict

METHOD = "context_patch_knn_v1"
MAX_CANVAS_PIXELS = 512 * 512
MAX_MASKED_PIXELS = 1024


@dataclass
class InpaintMetrics:
    filled_pixels: int
    mask_ratio: float
    verdict: InpaintRouteVerdict
    elapsed_ms: float
    method: str = METHOD
    comparisons: int = 0
    discarded_pixels: int = 0


@dataclass
class InpaintResult:
    image: Image.Image
    metrics: InpaintMetrics


def inpaint_patch_knn(
    image: Image.Image,
    mask: Image.Image,
    *,
    patch_size: int = 5,
    search_radius: int = 25,
    k_neighbors: int = 3,
    max_comparisons: int = 200_000,
    max_context_error: float = 0.05,
) -> InpaintResult:
    """Match visible context, then average RGB centers of the k best donors.

    Mask values >128 select pixels. Donor patches must be completely outside
    the original selection. Filled context can guide the next boundary layer,
    but generated pixels never become donor patches. Unsupported or partial
    repairs return the original image with filled_pixels=0.
    """
    started = time.perf_counter()
    if image.size != mask.size:
        raise ValueError("Image and defect mask dimensions must match")
    if image.mode not in ("RGB", "RGBA"):
        raise ValueError("Inpaint expects RGB or RGBA pixels")
    w, h = image.size
    if not w or not h or w * h > MAX_CANVAS_PIXELS:
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

    verdict = route_inpaint_request(mask)
    comparisons = 0
    filled = 0

    def result(reason: str | None = None, output: Image.Image | None = None) -> InpaintResult:
        routed = verdict if reason is None else InpaintRouteVerdict(
            "escalate_diffusion", 0.0, verdict.mask_ratio, reason)
        return InpaintResult(image.copy() if output is None else output,
                             InpaintMetrics(filled if output is not None else 0,
                                            verdict.mask_ratio, routed,
                                            round((time.perf_counter() - started) * 1000, 3),
                                            comparisons=comparisons,
                                            discarded_pixels=filled if output is None else 0))

    if verdict.method != "patch_knn":
        return result()
    selected = mask.convert("L").point(lambda value: 255 if value > 128 else 0)
    mask_px = selected.load()
    remaining = {(x, y) for y in range(h) for x in range(w) if mask_px[x, y]}
    if not remaining:
        return result()
    if len(remaining) > MAX_MASKED_PIXELS:
        return result("Selection exceeds the 1024-pixel preview limit")

    original = image.convert("RGBA")
    src = original.load()
    if any(src[x, y][3] == 0 for x, y in remaining):
        return result("Selected transparent pixels have no visible color context")
    output = original.copy()
    out = output.load()
    half = patch_size // 2
    offsets = [(dx, dy) for dy in range(-half, half + 1) for dx in range(-half, half + 1)
               if dx or dy]

    # Integral mask: each donor's entire original patch must be unselected.
    integral = [[0] * (w + 1) for _ in range(h + 1)]
    for y in range(h):
        running = 0
        for x in range(w):
            running += bool(mask_px[x, y])
            integral[y + 1][x + 1] = integral[y][x + 1] + running
    donors = set()
    for y in range(half, h - half):
        for x in range(half, w - half):
            left, top, right, bottom = x - half, y - half, x + half + 1, y + half + 1
            count = integral[bottom][right] - integral[top][right] - integral[bottom][left] + integral[top][left]
            if count == 0 and src[x, y][3] > 0:
                donors.add((x, y))
    if len(donors) < k_neighbors:
        return result("Too few intact original donor patches")

    search_offsets = sorted(((dx, dy) for dy in range(-search_radius, search_radius + 1)
                             for dx in range(-search_radius, search_radius + 1)),
                            key=lambda xy: (xy[0] ** 2 + xy[1] ** 2, xy[1], xy[0]))

    while remaining:
        frontier = []
        for x, y in sorted(remaining, key=lambda xy: (xy[1], xy[0])):
            context = [(dx, dy, out[x + dx, y + dy]) for dx, dy in offsets
                       if 0 <= x + dx < w and 0 <= y + dy < h
                       and (x + dx, y + dy) not in remaining and out[x + dx, y + dy][3] > 0]
            if context:
                frontier.append((x, y, context))
        if not frontier:
            return result("No visible boundary context for remaining pixels")
        # A layer uses only context available before that layer, independent of set order.
        for x, y, context in frontier:
            scores = []
            exact_matches = 0
            for sx, sy in search_offsets:
                cx, cy = x + sx, y + sy
                if (cx, cy) not in donors:
                    continue
                if comparisons >= max_comparisons:
                    return result("Context comparison budget exhausted; partial preview discarded")
                comparisons += 1
                error = 0.0
                for dx, dy, target in context:
                    donor = src[cx + dx, cy + dy]
                    visibility = min(target[3], donor[3]) / 255
                    error += visibility * sum((target[c] - donor[c]) ** 2 for c in range(3)) / 3
                    error += (target[3] - donor[3]) ** 2
                error /= len(context) * 255 ** 2
                if error <= max_context_error:
                    scores.append((error, sx ** 2 + sy ** 2, cy, cx))
                if error == 0:
                    exact_matches += 1
                    if exact_matches == k_neighbors:
                        # Zero is optimal. Spatially ordered traversal also settles all ties.
                        break
            nearest = heapq.nsmallest(k_neighbors, scores)
            if len(nearest) < k_neighbors:
                return result("Too few nearby donor patches match the visible context")
            rgb = tuple(round(sum(src[cx, cy][channel] for _, _, cy, cx in nearest) / k_neighbors)
                        for channel in range(3))
            out[x, y] = (*rgb, src[x, y][3])
            remaining.remove((x, y))
            filled += 1
    return result(output=output.convert(image.mode))
