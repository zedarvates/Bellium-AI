from __future__ import annotations

import math
from bellium.knn._image import Image, Rgb, clamp_rgb, shape

def _luma(p: Rgb) -> float:
    return (2126 * p[0] + 7152 * p[1] + 722 * p[2]) / 10000.0

def contrast_stretch(image: Image, black_percentile: float = 0.01, white_percentile: float = 0.99) -> Image:
    height, width = shape(image)
    lumas = sorted([_luma(px) for row in image for px in row])
    n = len(lumas)
    idx_low = max(0, min(n - 1, int(black_percentile * n)))
    idx_high = max(0, min(n - 1, int(white_percentile * n)))
    low_val = lumas[idx_low]
    high_val = lumas[idx_high]
    span = high_val - low_val
    if span < 1.0:
        return [list(r) for r in image]
    out = []
    for r in range(height):
        row = []
        for c in range(width):
            px = image[r][c]
            channels = []
            for ch in range(3):
                v = (px[ch] - low_val) * 255.0 / span
                channels.append(clamp_rgb(v))
            row.append((channels[0], channels[1], channels[2]))
        out.append(row)
    return out

def histogram_equalize(image: Image) -> Image:
    height, width = shape(image)
    hist = [0] * 256
    total_pixels = height * width
    for row in image:
        for px in row:
            luma_val = clamp_rgb(_luma(px))
            hist[luma_val] += 1
    cdf = [0] * 256
    acc = 0
    for i in range(256):
        acc += hist[i]
        cdf[i] = acc
    cdf_min = min(v for v in cdf if v > 0) if any(v > 0 for v in cdf) else 0
    lut = [0] * 256
    for i in range(256):
        if total_pixels > cdf_min:
            lut[i] = clamp_rgb((cdf[i] - cdf_min) * 255.0 / (total_pixels - cdf_min))
        else:
            lut[i] = i
    out = []
    for r in range(height):
        row = []
        for c in range(width):
            px = image[r][c]
            orig_luma = _luma(px)
            target_luma = lut[clamp_rgb(orig_luma)]
            if orig_luma > 0:
                ratio = target_luma / orig_luma
            else:
                ratio = 1.0
            row.append((clamp_rgb(px[0] * ratio), clamp_rgb(px[1] * ratio), clamp_rgb(px[2] * ratio)))
        out.append(row)
    return out

def auto_gamma(image: Image, target_luma: float = 128.0) -> Image:
    height, width = shape(image)
    lumas = [_luma(px) for row in image for px in row]
    mean_l = sum(lumas) / len(lumas)
    if mean_l <= 1.0 or mean_l >= 254.0:
        return [list(r) for r in image]
    gamma = math.log(target_luma / 255.0) / math.log(mean_l / 255.0)
    gamma = max(0.2, min(5.0, gamma))
    inv_gamma = 1.0 / gamma
    lut = [clamp_rgb(((i / 255.0) ** inv_gamma) * 255.0) for i in range(256)]
    out = []
    for r in range(height):
        row = []
        for c in range(width):
            px = image[r][c]
            row.append((lut[px[0]], lut[px[1]], lut[px[2]]))
        out.append(row)
    return out
