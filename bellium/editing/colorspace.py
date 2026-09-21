from __future__ import annotations

import math
from bellium.knn._image import Rgb, clamp_rgb

# Standard D65 illuminant white point for sRGB
D65_X = 0.95047
D65_Y = 1.00000
D65_Z = 1.08883

def rgb_to_hsv(rgb: Rgb) -> tuple[float, float, float]:
    r, g, b = rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0
    c_max = max(r, g, b)
    c_min = min(r, g, b)
    delta = c_max - c_min
    # Hue
    if delta == 0:
        h = 0.0
    elif c_max == r:
        h = (60.0 * ((g - b) / delta) + 360.0) % 360.0
    elif c_max == g:
        h = (60.0 * ((b - r) / delta) + 120.0) % 360.0
    else:
        h = (60.0 * ((r - g) / delta) + 240.0) % 360.0
    # Saturation
    s = 0.0 if c_max == 0 else delta / c_max
    # Value
    v = c_max
    return round(h, 2), round(s, 4), round(v, 4)

def hsv_to_rgb(h: float, s: float, v: float) -> Rgb:
    h = h % 360.0
    s = max(0.0, min(1.0, s))
    v = max(0.0, min(1.0, v))
    c = v * s
    x = c * (1.0 - abs(((h / 60.0) % 2) - 1.0))
    m = v - c
    if 0 <= h < 60:
        r1, g1, b1 = c, x, 0.0
    elif 60 <= h < 120:
        r1, g1, b1 = x, c, 0.0
    elif 120 <= h < 180:
        r1, g1, b1 = 0.0, c, x
    elif 180 <= h < 240:
        r1, g1, b1 = 0.0, x, c
    elif 240 <= h < 300:
        r1, g1, b1 = x, 0.0, c
    else:
        r1, g1, b1 = c, 0.0, x
    return clamp_rgb((r1 + m) * 255.0), clamp_rgb((g1 + m) * 255.0), clamp_rgb((b1 + m) * 255.0)

def _srgb_to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

def _linear_to_srgb(c: float) -> float:
    return 12.92 * c if c <= 0.0031308 else 1.055 * (c ** (1.0 / 2.4)) - 0.055

def rgb_to_xyz(rgb: Rgb) -> tuple[float, float, float]:
    r = _srgb_to_linear(rgb[0] / 255.0)
    g = _srgb_to_linear(rgb[1] / 255.0)
    b = _srgb_to_linear(rgb[2] / 255.0)
    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041
    return x, y, z

def _lab_f(t: float) -> float:
    delta = 6.0 / 29.0
    return t ** (1.0 / 3.0) if t > delta ** 3 else (t / (3.0 * delta * delta)) + (4.0 / 29.0)

def xyz_to_lab(x: float, y: float, z: float) -> tuple[float, float, float]:
    fx = _lab_f(x / D65_X)
    fy = _lab_f(y / D65_Y)
    fz = _lab_f(z / D65_Z)
    l_val = max(0.0, min(100.0, 116.0 * fy - 16.0))
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return round(l_val, 3), round(a, 3), round(b, 3)

def rgb_to_lab(rgb: Rgb) -> tuple[float, float, float]:
    x, y, z = rgb_to_xyz(rgb)
    return xyz_to_lab(x, y, z)

def delta_e_cie76(c1: Rgb, c2: Rgb) -> float:
    """CIE 1976 Euclidean color distance in L*a*b* space."""
    l1, a1, b1 = rgb_to_lab(c1)
    l2, a2, b2 = rgb_to_lab(c2)
    dl = l1 - l2
    da = a1 - a2
    db = b1 - b2
    return math.sqrt(dl * dl + da * da + db * db)
