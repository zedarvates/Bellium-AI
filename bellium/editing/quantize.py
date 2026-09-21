from __future__ import annotations

from bellium.knn._image import Image, Rgb, clamp_rgb, shape

def _perceptual_dist_sq(c1: Rgb, c2: Rgb) -> float:
    dr = c1[0] - c2[0]
    dg = c1[1] - c2[1]
    db = c1[2] - c2[2]
    return 0.299 * dr * dr + 0.587 * dg * dg + 0.114 * db * db

def extract_palette_median_cut(image: Image, max_colors: int = 16) -> list[Rgb]:
    shape(image)
    if isinstance(max_colors, bool) or not isinstance(max_colors, int) or max_colors <= 0:
        raise ValueError('max_colors must be a positive integer')
    unique_pixels = list({px for row in image for px in row})
    if len(unique_pixels) <= max_colors:
        return sorted(unique_pixels)
    buckets = [unique_pixels]
    while len(buckets) < max_colors:
        best_bucket_idx = -1
        max_range = -1
        split_channel = 0
        for idx, b in enumerate(buckets):
            if len(b) <= 1:
                continue
            for ch in range(3):
                vals = [p[ch] for p in b]
                rng = max(vals) - min(vals)
                if rng > max_range:
                    max_range = rng
                    best_bucket_idx = idx
                    split_channel = ch
        if best_bucket_idx == -1 or max_range == 0:
            break
        target_b = buckets.pop(best_bucket_idx)
        target_b.sort(key=lambda p: p[split_channel])
        mid = len(target_b) // 2
        buckets.append(target_b[:mid])
        buckets.append(target_b[mid:])
    palette = []
    for b in buckets:
        if not b:
            continue
        r_avg = sum(p[0] for p in b) / len(b)
        g_avg = sum(p[1] for p in b) / len(b)
        b_avg = sum(p[2] for p in b) / len(b)
        palette.append((clamp_rgb(r_avg), clamp_rgb(g_avg), clamp_rgb(b_avg)))
    return palette

def quantize_nearest(image: Image, palette: list[Rgb]) -> Image:
    if not palette:
        raise ValueError('palette must contain at least one color')
    height, width = shape(image)
    out = []
    for r in range(height):
        row = []
        for c in range(width):
            px = image[r][c]
            best_color = min(palette, key=lambda pal: _perceptual_dist_sq(px, pal))
            row.append(best_color)
        out.append(row)
    return out

def quantize_floyd_steinberg(image: Image, palette: list[Rgb], dither_strength: float = 1.0) -> Image:
    if not palette:
        raise ValueError('palette must contain at least one color')
    height, width = shape(image)
    strength = max(0.0, min(1.0, float(dither_strength)))
    work = [[list(px) for px in row] for row in image]
    out = []
    for r in range(height):
        out_row = []
        for c in range(width):
            curr = (clamp_rgb(work[r][c][0]), clamp_rgb(work[r][c][1]), clamp_rgb(work[r][c][2]))
            nearest = min(palette, key=lambda pal: _perceptual_dist_sq(curr, pal))
            out_row.append(nearest)
            err_r = (curr[0] - nearest[0]) * strength
            err_g = (curr[1] - nearest[1]) * strength
            err_b = (curr[2] - nearest[2]) * strength
            if c + 1 < width:
                work[r][c + 1][0] += err_r * 7.0 / 16.0
                work[r][c + 1][1] += err_g * 7.0 / 16.0
                work[r][c + 1][2] += err_b * 7.0 / 16.0
            if r + 1 < height:
                if c > 0:
                    work[r + 1][c - 1][0] += err_r * 3.0 / 16.0
                    work[r + 1][c - 1][1] += err_g * 3.0 / 16.0
                    work[r + 1][c - 1][2] += err_b * 3.0 / 16.0
                work[r + 1][c][0] += err_r * 5.0 / 16.0
                work[r + 1][c][1] += err_g * 5.0 / 16.0
                work[r + 1][c][2] += err_b * 5.0 / 16.0
                if c + 1 < width:
                    work[r + 1][c + 1][0] += err_r * 1.0 / 16.0
                    work[r + 1][c + 1][1] += err_g * 1.0 / 16.0
                    work[r + 1][c + 1][2] += err_b * 1.0 / 16.0
        out.append(out_row)
    return out

BAYER_4X4 = [
    [ 0,  8,  2, 10],
    [12,  4, 14,  6],
    [ 3, 11,  1,  9],
    [15,  7, 13,  5],
]

def quantize_ordered(image: Image, palette: list[Rgb], matrix_size: int = 4) -> Image:
    if not palette:
        raise ValueError('palette must contain at least one color')
    height, width = shape(image)
    out = []
    for r in range(height):
        row = []
        for c in range(width):
            px = image[r][c]
            bayer_val = (BAYER_4X4[r % 4][c % 4] / 16.0 - 0.5) * 32.0
            dithered_px = (
                clamp_rgb(px[0] + bayer_val),
                clamp_rgb(px[1] + bayer_val),
                clamp_rgb(px[2] + bayer_val),
            )
            nearest = min(palette, key=lambda pal: _perceptual_dist_sq(dithered_px, pal))
            row.append(nearest)
        out.append(row)
    return out
