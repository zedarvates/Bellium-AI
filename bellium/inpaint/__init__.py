"""Bellium Inpainting: Patch k-NN synthesis, deterministic hole-filling and escalation routing."""

from .patch_knn import inpaint_patch_knn, InpaintResult, InpaintMetrics
from .router import route_inpaint_request, InpaintRouteVerdict

__all__ = [
    "inpaint_patch_knn",
    "InpaintResult",
    "InpaintMetrics",
    "route_inpaint_request",
    "InpaintRouteVerdict",
]
