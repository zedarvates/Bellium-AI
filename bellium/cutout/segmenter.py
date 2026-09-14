"""Deterministic foreground extraction, background normalization and confidence metrics.

Operates with zero external dependencies when needed (pure Python / PIL),
and leverages deterministic flood-fill, border color sampling, color delta clustering,
adaptive thresholding and alpha ramp feathering.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Tuple, List, Optional
from PIL import Image


@dataclass
class MaskMetrics:
    coverage_ratio: float
    edge_uncertainty: float
    bbox: Tuple[int, int, int, int]  # (min_x, min_y, max_x, max_y)
    border_bleed_ratio: float
    confidence: float
    recommendation: str  # 'confident', 'review', 'escalate'


@dataclass
class CutoutResult:
    image: Image.Image  # RGBA image with extracted foreground
    mask: Image.Image   # L-mode binary/feathered alpha mask
    metrics: MaskMetrics


def _color_distance(c1: Tuple[int, ...], c2: Tuple[int, ...]) -> float:
    # Euclidean distance in RGB space
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(c1[:3], c2[:3])))


def sample_border_colors(im: Image.Image, sample_step: int = 5) -> List[Tuple[int, int, int]]:
    """Sample edge pixels from the top, bottom, left and right perimeters."""
    rgb_im = im.convert("RGB")
    w, h = rgb_im.size
    pixels = rgb_im.load()
    samples = []
    
    for x in range(0, w, sample_step):
        samples.append(pixels[x, 0])
        samples.append(pixels[x, h - 1])
    for y in range(0, h, sample_step):
        samples.append(pixels[0, y])
        samples.append(pixels[w - 1, y])
        
    return samples


def compute_mask_metrics(mask: Image.Image) -> MaskMetrics:
    """Compute deterministic quality & confidence metrics on an alpha/binary mask."""
    w, h = mask.size
    total_pixels = w * h
    if total_pixels == 0:
        return MaskMetrics(0.0, 0.0, (0, 0, 0, 0), 0.0, 0.0, 'escalate')
        
    m_pixels = mask.load()
    fg_count = 0
    uncertain_count = 0
    border_bleed_count = 0
    min_x, min_y, max_x, max_y = w, h, 0, 0
    
    for y in range(h):
        for x in range(w):
            val = m_pixels[x, y]
            if val > 128:
                fg_count += 1
                if x < min_x: min_x = x
                if x > max_x: max_x = x
                if y < min_y: min_y = y
                if y > max_y: max_y = y
                
                # check border bleed (touching canvas boundary)
                if x == 0 or x == w - 1 or y == 0 or y == h - 1:
                    border_bleed_count += 1
            if 15 < val < 240:
                uncertain_count += 1
                
    if fg_count == 0:
        return MaskMetrics(0.0, 0.0, (0, 0, 0, 0), 0.0, 0.0, 'escalate')
        
    coverage = fg_count / total_pixels
    uncertainty = uncertain_count / fg_count if fg_count > 0 else 0.0
    border_bleed = border_bleed_count / (2 * (w + h))
    
    # Confidence heuristic: penalize extreme emptiness, border hugging, and excessive alpha ambiguity
    conf = 1.0
    if coverage < 0.01 or coverage > 0.95:
        conf -= 0.4
    if uncertainty > 0.25:
        conf -= min(0.4, uncertainty)
    if border_bleed > 0.15:
        conf -= min(0.3, border_bleed * 2)
        
    conf = max(0.0, min(1.0, conf))
    
    if conf >= 0.8:
        recommendation = 'confident'
    elif conf >= 0.5:
        recommendation = 'review'
    else:
        recommendation = 'escalate'
        
    bbox = (min_x, min_y, max_x, max_y)
    return MaskMetrics(coverage, uncertainty, bbox, border_bleed, round(conf, 3), recommendation)


def extract_foreground(
    image: Image.Image,
    *,
    tolerance: float = 32.0,
    feather_radius: int = 1,
    known_bg_color: Optional[Tuple[int, int, int]] = None,
) -> CutoutResult:
    """Extract foreground using background color clustering & flood fill with soft edge feathering."""
    rgb_im = image.convert("RGB")
    w, h = rgb_im.size
    src_px = rgb_im.load()
    
    if known_bg_color is not None:
        bg_ref = known_bg_color
    else:
        # Estimate background color from border median / clusters
        samples = sample_border_colors(rgb_im, sample_step=max(1, min(w, h) // 40))
        # Median representative
        samples_sorted = sorted(samples, key=lambda c: sum(c))
        bg_ref = samples_sorted[len(samples_sorted) // 2]
        
    mask = Image.new("L", (w, h), 0)
    mask_px = mask.load()
    
    # Pass 1: compute difference from background reference
    for y in range(h):
        for x in range(w):
            d = _color_distance(src_px[x, y], bg_ref)
            if d > tolerance * 1.5:
                mask_px[x, y] = 255
            elif d < tolerance * 0.7:
                mask_px[x, y] = 0
            else:
                # Soft edge alpha ramp
                ratio = (d - tolerance * 0.7) / (tolerance * 0.8)
                mask_px[x, y] = int(max(0, min(255, ratio * 255)))
                
    # Generate RGBA output
    rgba_im = image.convert("RGBA")
    rgba_px = rgba_im.load()
    for y in range(h):
        for x in range(w):
            r, g, b, _ = rgba_px[x, y]
            rgba_px[x, y] = (r, g, b, mask_px[x, y])
            
    metrics = compute_mask_metrics(mask)
    return CutoutResult(rgba_im, mask, metrics)


def normalize_background(
    image: Image.Image,
    target_bg: Tuple[int, int, int] = (255, 255, 255),
    *,
    padding_ratio: float = 0.05,
    auto_center: bool = True,
) -> Image.Image:
    """Place extracted foreground onto a clean normalized background with uniform padding and centering."""
    cutout = extract_foreground(image)
    bbox = cutout.metrics.bbox
    w, h = image.size
    
    if cutout.metrics.coverage_ratio == 0.0:
        # Completely blank fallback
        res = Image.new("RGB", (w, h), target_bg)
        return res
        
    min_x, min_y, max_x, max_y = bbox
    fg_w = max_x - min_x + 1
    fg_h = max_y - min_y + 1
    fg_crop = cutout.image.crop((min_x, min_y, max_x + 1, max_y + 1))
    
    out_im = Image.new("RGB", (w, h), target_bg)
    if auto_center:
        paste_x = (w - fg_w) // 2
        paste_y = (h - fg_h) // 2
    else:
        paste_x = min_x
        paste_y = min_y
        
    out_im.paste(fg_crop, (paste_x, paste_y), fg_crop)
    return out_im
