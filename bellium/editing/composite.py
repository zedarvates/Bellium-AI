from __future__ import annotations
from bellium.knn._image import Image, Mask, Rgb, clamp_rgb, shape

BLEND_MODES = (
    'over',
    'multiply',
    'screen',
    'overlay',
    'darken',
    'lighten',
    'difference',
    'add',
    'subtract',
    'color_dodge',
    'soft_light',
)

def _blend_pixel(cb: Rgb, cs: Rgb, mode: str) -> tuple[float, float, float]:
    rb, gb, bb = cb[0] / 255.0, cb[1] / 255.0, cb[2] / 255.0
    rs, gs, bs = cs[0] / 255.0, cs[1] / 255.0, cs[2] / 255.0
    
    def blend_ch(b: float, s: float) -> float:
        if mode == 'over':
            return s
        elif mode == 'multiply':
            return b * s
        elif mode == 'screen':
            return 1.0 - (1.0 - b) * (1.0 - s)
        elif mode == 'overlay':
            return (2.0 * b * s) if b < 0.5 else (1.0 - 2.0 * (1.0 - b) * (1.0 - s))
        elif mode == 'darken':
            return min(b, s)
        elif mode == 'lighten':
            return max(b, s)
        elif mode == 'difference':
            return abs(b - s)
        elif mode == 'add':
            return min(1.0, b + s)
        elif mode == 'subtract':
            return max(0.0, b - s)
        elif mode == 'color_dodge':
            return 1.0 if s >= 1.0 else min(1.0, b / (1.0 - s + 1e-6))
        elif mode == 'soft_light':
            return (1.0 - 2.0 * s) * b * b + 2.0 * s * b
        return s
        
    return (blend_ch(rb, rs) * 255.0, blend_ch(gb, gs) * 255.0, blend_ch(bb, bs) * 255.0)

def composite(
    base: Image,
    overlay: Image,
    *,
    mode: str = 'over',
    opacity: float = 1.0,
    mask: Mask | None = None,
    offset: tuple[int, int] = (0, 0),
) -> Image:
    b_h, b_w = shape(base)
    o_h, o_w = shape(overlay)
    if mode not in BLEND_MODES:
        known = ', '.join(BLEND_MODES)
        raise ValueError(f'unknown blend mode {mode}; known: {known}')
    opac = max(0.0, min(1.0, float(opacity)))
    off_y, off_x = offset
    out = [[list(px) for px in row] for row in base]
    for oy in range(o_h):
        by = oy + off_y
        if not (0 <= by < b_h):
            continue
        for ox in range(o_w):
            bx = ox + off_x
            if not (0 <= bx < b_w):
                continue
            p_base = base[by][bx]
            p_over = overlay[oy][ox]
            m_val = 1.0 if mask is None else (1.0 if mask[oy][ox] else 0.0)
            eff_alpha = opac * m_val
            if eff_alpha <= 0.0:
                continue
            blended = _blend_pixel(p_base, p_over, mode)
            out_r = clamp_rgb(p_base[0] + (blended[0] - p_base[0]) * eff_alpha)
            out_g = clamp_rgb(p_base[1] + (blended[1] - p_base[1]) * eff_alpha)
            out_b = clamp_rgb(p_base[2] + (blended[2] - p_base[2]) * eff_alpha)
            out[by][bx] = (out_r, out_g, out_b)
    return out
