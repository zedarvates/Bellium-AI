from __future__ import annotations

import math
from bellium.knn._image import Image, Rgb, clamp_rgb, shape

def _luma(p: Rgb) -> float:
    return (2126 * p[0] + 7152 * p[1] + 722 * p[2]) / 10000.0

def compare_images(image_a: Image, image_b: Image) -> dict[str, float]:
    h_a, w_a = shape(image_a)
    h_b, w_b = shape(image_b)
    if h_a != h_b or w_a != w_b:
        raise ValueError(f'image dimensions mismatch: {h_a}x{w_a} vs {h_b}x{w_b}')
    total_pixels = h_a * w_a
    total_abs_diff = 0.0
    total_sq_diff = 0.0
    max_channel_delta = 0
    mismatch_count = 0
    sum_a = sum_b = 0.0
    sum_sq_a = sum_sq_b = sum_ab = 0.0
    for r in range(h_a):
        for c in range(w_a):
            pa, pb = image_a[r][c], image_b[r][c]
            diff_r = abs(pa[0] - pb[0])
            diff_g = abs(pa[1] - pb[1])
            diff_b = abs(pa[2] - pb[2])
            max_delta = max(diff_r, diff_g, diff_b)
            if max_delta > 0:
                mismatch_count += 1
            if max_delta > max_channel_delta:
                max_channel_delta = max_delta
            abs_diff = (diff_r + diff_g + diff_b) / 3.0
            sq_diff = (diff_r * diff_r + diff_g * diff_g + diff_b * diff_b) / 3.0
            total_abs_diff += abs_diff
            total_sq_diff += sq_diff
            la, lb = _luma(pa), _luma(pb)
            sum_a += la
            sum_b += lb
            sum_sq_a += la * la
            sum_sq_b += lb * lb
            sum_ab += la * lb
    mae = total_abs_diff / total_pixels
    mse = total_sq_diff / total_pixels
    rmse = math.sqrt(mse)
    if mse < 1e-10:
        psnr = float('inf')
    else:
        psnr = 20.0 * math.log10(255.0 / rmse)
    mu_a = sum_a / total_pixels
    mu_b = sum_b / total_pixels
    var_a = (sum_sq_a / total_pixels) - mu_a * mu_a
    var_b = (sum_sq_b / total_pixels) - mu_b * mu_b
    cov_ab = (sum_ab / total_pixels) - mu_a * mu_b
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    num = (2.0 * mu_a * mu_b + c1) * (2.0 * cov_ab + c2)
    den = (mu_a * mu_a + mu_b * mu_b + c1) * (var_a + var_b + c2)
    ssim = num / den if den != 0.0 else 1.0
    return {
        'mae': round(mae, 4),
        'rmse': round(rmse, 4),
        'psnr': round(psnr, 2) if math.isfinite(psnr) else 999.0,
        'ssim': round(ssim, 4),
        'max_channel_delta': float(max_channel_delta),
        'mismatch_pixels': float(mismatch_count),
        'mismatch_ratio': round(mismatch_count / total_pixels, 4),
    }

def diff_image(image_a: Image, image_b: Image, highlight: Rgb = (255, 0, 0)) -> Image:
    h_a, w_a = shape(image_a)
    h_b, w_b = shape(image_b)
    if h_a != h_b or w_a != w_b:
        raise ValueError('image dimensions must match to compute diff')
    out = []
    for r in range(h_a):
        row = []
        for c in range(w_a):
            pa, pb = image_a[r][c], image_b[r][c]
            if pa != pb:
                row.append(highlight)
            else:
                gray = clamp_rgb(_luma(pa) * 0.4)
                row.append((gray, gray, gray))
        out.append(row)
    return out
