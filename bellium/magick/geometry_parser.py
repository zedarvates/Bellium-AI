from __future__ import annotations

import re

def parse_geometry(
    spec: str,
    orig_w: int,
    orig_h: int,
) -> tuple[int, int, tuple[int, int, int, int] | None]:
    """Parses ImageMagick geometry specification and returns (target_w, target_h, crop_bbox)."""
    s = spec.strip()
    # Percentages like 50% or 50%x75%
    pct_match = re.match(r'^(\d+)%(?:x(\d+)%)?$', s)
    if pct_match:
        pct_x = int(pct_match.group(1)) / 100.0
        pct_y = int(pct_match.group(2)) / 100.0 if pct_match.group(2) else pct_x
        return max(1, int(round(orig_w * pct_x))), max(1, int(round(orig_h * pct_y))), None
        
    # Dimension with optional flags: 800x600, 800x600!, 800x600^, 800x600#, 800x600>, 800x600<
    dim_match = re.match(r'^(\d+)?x(\d+)?([!\^\#><])?$', s)
    if not dim_match:
        raise ValueError(f'invalid geometry specification: {spec}')
        
    w_str, h_str, flag = dim_match.groups()
    box_w = int(w_str) if w_str else None
    box_h = int(h_str) if h_str else None
    
    if box_w is None and box_h is None:
        return orig_w, orig_h, None
    if box_w is None:
        assert box_h is not None
        scale = box_h / orig_h
        return max(1, int(round(orig_w * scale))), box_h, None
    if box_h is None:
        scale = box_w / orig_w
        return box_w, max(1, int(round(orig_h * scale))), None
        
    if flag == '!':  # Exact forced resize
        return box_w, box_h, None
    elif flag == '^':  # Minimum cover
        scale = max(box_w / orig_w, box_h / orig_h)
        return max(1, int(round(orig_w * scale))), max(1, int(round(orig_h * scale))), None
    elif flag == '#':  # Crop to fit (scale minimum then center crop)
        scale = max(box_w / orig_w, box_h / orig_h)
        scaled_w = max(1, int(round(orig_w * scale)))
        scaled_h = max(1, int(round(orig_h * scale)))
        crop_x = (scaled_w - box_w) // 2
        crop_y = (scaled_h - box_h) // 2
        return scaled_w, scaled_h, (crop_y, crop_x, crop_y + box_h, crop_x + box_w)
    elif flag == '>':  # Only scale down if larger
        if orig_w > box_w or orig_h > box_h:
            scale = min(box_w / orig_w, box_h / orig_h)
            return max(1, int(round(orig_w * scale))), max(1, int(round(orig_h * scale))), None
        return orig_w, orig_h, None
    elif flag == '<':  # Only scale up if smaller
        if orig_w < box_w and orig_h < box_h:
            scale = min(box_w / orig_w, box_h / orig_h)
            return max(1, int(round(orig_w * scale))), max(1, int(round(orig_h * scale))), None
        return orig_w, orig_h, None
    else:  # Default: scale to fit within bounding box
        scale = min(box_w / orig_w, box_h / orig_h)
        return max(1, int(round(orig_w * scale))), max(1, int(round(orig_h * scale))), None
