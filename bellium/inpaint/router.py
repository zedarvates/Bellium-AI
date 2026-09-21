"""Deterministic routing between instant patch synthesis and heavier model escalation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from PIL import Image


@dataclass
class InpaintRouteVerdict:
    method: str  # 'patch_knn', 'micro_inpaint', 'escalate_diffusion'
    confidence: float | None
    mask_ratio: float
    reason: str


def route_inpaint_request(mask: Image.Image, max_local_ratio: float = 0.12) -> InpaintRouteVerdict:
    """Analyze mask size and topological spread to decide if fast local synthesis is competent."""
    if (
        isinstance(max_local_ratio, bool)
        or not isinstance(max_local_ratio, (int, float))
        or not math.isfinite(max_local_ratio)
        or not 0 < max_local_ratio <= 0.12
    ):
        raise ValueError("max_local_ratio must be positive and at most 0.12")
    gray = mask.convert("L")
    w, h = gray.size
    total = w * h
    if total == 0:
        return InpaintRouteVerdict("escalate_diffusion", 0.0, 0.0, "Empty canvas")

    px = gray.load()
    hole_count = 0
    for y in range(h):
        for x in range(w):
            if px[x, y] > 128:
                hole_count += 1

    ratio = hole_count / total
    bbox = gray.point(lambda v: 255 if v > 128 else 0).getbbox()
    if bbox and max(bbox[2] - bbox[0], bbox[3] - bbox[1]) > max(w, h) * 0.45:
        return InpaintRouteVerdict(
            "escalate_diffusion", 0.0, ratio, "Mask span exceeds patch competence"
        )
    if ratio == 0.0:
        return InpaintRouteVerdict("patch_knn", 1.0, 0.0, "No mask: identity pass")
    elif ratio <= 0.05:
        return InpaintRouteVerdict(
            "patch_knn", 0.95, ratio, "Small bounded mask: instant patch k-NN synthesis"
        )
    elif ratio <= max_local_ratio:
        return InpaintRouteVerdict(
            "patch_knn", 0.80, ratio, "Moderate mask: patch k-NN suitable with texture context"
        )
    else:
        return InpaintRouteVerdict(
            "escalate_diffusion",
            0.40,
            ratio,
            f"Large mask ({ratio:.1%}): exceeds local exemplar competence, escalate to generative model",
        )
