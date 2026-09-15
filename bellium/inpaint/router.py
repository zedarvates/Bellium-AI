"""Deterministic routing between instant patch synthesis and heavier model escalation."""
from __future__ import annotations

from dataclasses import dataclass
from PIL import Image


@dataclass
class InpaintRouteVerdict:
    method: str  # 'patch_knn', 'micro_inpaint', 'escalate_diffusion'
    confidence: float
    mask_ratio: float
    reason: str


def route_inpaint_request(mask: Image.Image, max_local_ratio: float = 0.25) -> InpaintRouteVerdict:
    """Analyze mask size and topological spread to decide if fast local synthesis is competent."""
    gray = mask.convert("L")
    w, h = gray.size
    total = w * h
    if total == 0:
        return InpaintRouteVerdict('escalate_diffusion', 0.0, 0.0, 'Empty canvas')
        
    px = gray.load()
    hole_count = 0
    for y in range(h):
        for x in range(w):
            if px[x, y] > 128:
                hole_count += 1
                
    ratio = hole_count / total
    if ratio == 0.0:
        return InpaintRouteVerdict('patch_knn', 1.0, 0.0, 'No mask: identity pass')
    elif ratio <= 0.05:
        return InpaintRouteVerdict('patch_knn', 0.95, ratio, 'Small bounded mask: instant patch k-NN synthesis')
    elif ratio <= max_local_ratio:
        return InpaintRouteVerdict('patch_knn', 0.80, ratio, 'Moderate mask: patch k-NN suitable with texture context')
    else:
        return InpaintRouteVerdict('escalate_diffusion', 0.40, ratio, f'Large mask ({ratio:.1%}): exceeds local exemplar competence, escalate to generative model')
