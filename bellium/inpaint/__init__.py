"""Bellium Inpainting: Patch k-NN synthesis, deterministic hole-filling and escalation routing."""
from .patch_knn import inpaint_patch_knn, InpaintResult, InpaintMetrics
from .preview import inpaint_preview, PreviewMetrics, InterpolationGate, PREVIEW_METHODS
from .router import route_inpaint_request, InpaintRouteVerdict

__all__ = [
    "inpaint_patch_knn",
    "inpaint_preview",
    "PreviewMetrics",
    "InterpolationGate",
    "PREVIEW_METHODS",
    "InpaintResult",
    "InpaintMetrics",
    "route_inpaint_request",
    "InpaintRouteVerdict",
]
