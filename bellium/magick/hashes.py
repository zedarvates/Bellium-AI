from __future__ import annotations

import math
from bellium.editing.resample import resample_box
from bellium.knn._image import Image, Rgb

def _luma(p: Rgb) -> float:
    return (2126 * p[0] + 7152 * p[1] + 722 * p[2]) / 10000.0

def compute_ahash(image: Image) -> int:
    tiny = resample_box(image, 8, 8)
    lumas = [_luma(px) for row in tiny for px in row]
    avg = sum(lumas) / len(lumas)
    hash_val = 0
    for i, v in enumerate(lumas):
        if v >= avg:
            hash_val |= (1 << (63 - i))
    return hash_val

def compute_dhash(image: Image) -> int:
    tiny = resample_box(image, 8, 9)
    hash_val = 0
    bit_idx = 0
    for r in range(8):
        for c in range(8):
            left = _luma(tiny[r][c])
            right = _luma(tiny[r][c + 1])
            if left > right:
                hash_val |= (1 << (63 - bit_idx))
            bit_idx += 1
    return hash_val

def compute_phash(image: Image) -> int:
    tiny = resample_box(image, 32, 32)
    matrix = [[_luma(px) for px in row] for row in tiny]
    dct = [[0.0 for _ in range(8)] for _ in range(8)]
    pi_over_64 = math.pi / 64.0
    for u in range(8):
        cu = 1.0 / math.sqrt(2.0) if u == 0 else 1.0
        for v in range(8):
            cv = 1.0 / math.sqrt(2.0) if v == 0 else 1.0
            acc = 0.0
            for x in range(32):
                cos_x = math.cos((2 * x + 1) * u * pi_over_64)
                for y in range(32):
                    cos_y = math.cos((2 * y + 1) * v * pi_over_64)
                    acc += matrix[x][y] * cos_x * cos_y
            dct[u][v] = 0.25 * cu * cv * acc
    # Take 8x8 low frequencies excluding DC (0, 0)
    coeffs = []
    for u in range(8):
        for v in range(8):
            if u == 0 and v == 0:
                continue
            coeffs.append(dct[u][v])
    median_val = sorted(coeffs)[len(coeffs) // 2]
    hash_val = 0
    for i, c_val in enumerate(coeffs):
        if c_val > median_val:
            hash_val |= (1 << (62 - i))
    return hash_val

def hamming_distance(h1: int, h2: int) -> int:
    return bin(h1 ^ h2).count('1')

def hash_to_hex(h: int) -> str:
    return f'{h:016x}'
