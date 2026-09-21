"""Pillow adapter to Bellium's bounded, deterministic patch implementation."""

from __future__ import annotations

from dataclasses import dataclass
import time
from PIL import Image

from bellium.knn.patch_inpaint import inpaint
from .router import route_inpaint_request, InpaintRouteVerdict


@dataclass
class InpaintMetrics:
    filled_pixels: int
    mask_ratio: float
    verdict: InpaintRouteVerdict
    elapsed_ms: float
    quality: dict | None = None


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
) -> InpaintResult:
    """Fill a small mask, or return the unchanged image with an escalation verdict."""
    if image.size != mask.size:
        raise ValueError("mask shape must match image")
    start = time.perf_counter()
    verdict = route_inpaint_request(mask)
    rgb = image.convert("RGB")
    output = rgb.copy()
    if verdict.method == "escalate_diffusion":
        return InpaintResult(output, InpaintMetrics(0, verdict.mask_ratio, verdict, 0.0))
    width, height = rgb.size
    pixels = list(rgb.get_flattened_data() if hasattr(rgb, "get_flattened_data") else rgb.getdata())
    gray_mask = mask.convert("L")
    mask_pixels = list(gray_mask.get_flattened_data() if hasattr(gray_mask, "get_flattened_data") else gray_mask.getdata())
    rows = [pixels[r * width : (r + 1) * width] for r in range(height)]
    binary = [
        [int(v > 128) for v in mask_pixels[r * width : (r + 1) * width]] for r in range(height)
    ]
    result = inpaint(
        rows, binary, patch_size=patch_size, search_radius=search_radius, k_neighbors=k_neighbors
    )
    if result.abstained:
        verdict = InpaintRouteVerdict(
            "escalate_diffusion",
            result.confidence,
            verdict.mask_ratio,
            result.output.get("reason", "Patch support insufficient; result not applied"),
        )
        applied = 0
    else:
        output.putdata([pixel for row in result.output["image"] for pixel in row])
        applied = result.output["filled"]
        if result.output.get("quality") is not None:
            verdict = InpaintRouteVerdict(
                "patch_knn", None, verdict.mask_ratio,
                "Known-context checks passed; reconstruction probability is not calibrated",
            )
    return InpaintResult(
        output,
        InpaintMetrics(
            applied,
            verdict.mask_ratio,
            verdict,
            round((time.perf_counter() - start) * 1000, 2),
            result.output.get("quality"),
        ),
    )
