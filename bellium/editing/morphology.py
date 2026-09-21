from __future__ import annotations

from bellium.knn._image import Image, Mask, Rgb, clamp_rgb, shape

def _luma(p: Rgb) -> float:
    return (2126 * p[0] + 7152 * p[1] + 722 * p[2]) / 10000.0

def otsu_threshold(image: Image) -> tuple[Mask, int]:
    height, width = shape(image)
    total = height * width
    hist = [0] * 256
    for row in image:
        for px in row:
            luma_val = clamp_rgb(_luma(px))
            hist[luma_val] += 1
    sum_total = sum(i * hist[i] for i in range(256))
    sum_b = 0.0
    w_b = 0
    max_var = 0.0
    best_thresh = 128
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        between_var = w_b * w_f * ((m_b - m_f) ** 2)
        if between_var > max_var:
            max_var = between_var
            best_thresh = t
    mask = []
    for row in image:
        mask_row = [1 if _luma(px) >= best_thresh else 0 for px in row]
        mask.append(mask_row)
    return mask, best_thresh

def adaptive_local_threshold(image: Image, window_size: int = 5, c_offset: float = 5.0) -> Mask:
    height, width = shape(image)
    pad = max(1, window_size // 2)
    mask = []
    for r in range(height):
        mask_row = []
        for c in range(width):
            y_min, y_max = max(0, r - pad), min(height, r + pad + 1)
            x_min, x_max = max(0, c - pad), min(width, c + pad + 1)
            lumas = [_luma(image[y][x]) for y in range(y_min, y_max) for x in range(x_min, x_max)]
            local_mean = sum(lumas) / len(lumas)
            center = _luma(image[r][c])
            mask_row.append(1 if center >= (local_mean - c_offset) else 0)
        mask.append(mask_row)
    return mask

def dilate_mask(mask: Mask, radius: int = 1) -> Mask:
    height = len(mask)
    width = len(mask[0])
    out = []
    for r in range(height):
        row = []
        for c in range(width):
            y_min, y_max = max(0, r - radius), min(height, r + radius + 1)
            x_min, x_max = max(0, c - radius), min(width, c + radius + 1)
            val = 1 if any(mask[y][x] for y in range(y_min, y_max) for x in range(x_min, x_max)) else 0
            row.append(val)
        out.append(row)
    return out

def erode_mask(mask: Mask, radius: int = 1) -> Mask:
    height = len(mask)
    width = len(mask[0])
    out = []
    for r in range(height):
        row = []
        for c in range(width):
            y_min, y_max = max(0, r - radius), min(height, r + radius + 1)
            x_min, x_max = max(0, c - radius), min(width, c + radius + 1)
            val = 1 if all(mask[y][x] for y in range(y_min, y_max) for x in range(x_min, x_max)) else 0
            row.append(val)
        out.append(row)
    return out

def open_mask(mask: Mask, radius: int = 1) -> Mask:
    return dilate_mask(erode_mask(mask, radius), radius)

def close_mask(mask: Mask, radius: int = 1) -> Mask:
    return erode_mask(dilate_mask(mask, radius), radius)
