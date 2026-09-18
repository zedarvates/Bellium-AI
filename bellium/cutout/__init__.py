"""Bellium Cutout & Normalizer: deterministic CV + adaptive edge refinement."""

from .segmenter import (
    extract_foreground,
    normalize_background,
    compute_mask_metrics,
    CutoutResult,
    MaskMetrics,
    UnsafeCutoutError,
)

__all__ = [
    "extract_foreground",
    "normalize_background",
    "compute_mask_metrics",
    "CutoutResult",
    "MaskMetrics",
    "UnsafeCutoutError",
]
