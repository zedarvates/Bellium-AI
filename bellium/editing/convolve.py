from __future__ import annotations

import math
from bellium.knn._image import Image, clamp_rgb, shape

def convolve_2d(
    image: Image,
    kernel: list[list[float]],
    *,
    divisor: float | None = None,
    offset: float = 0.0,
) -> Image:
    height, width = shape(image)
    k_h = len(kernel)
    if k_h == 0 or len(kernel[0]) == 0 or k_h % 2 == 0 or len(kernel[0]) % 2 == 0:
        raise ValueError('kernel dimensions must be non-empty odd integers')
    k_w = len(kernel[0])
    pad_y = k_h // 2
    pad_x = k_w // 2
    div = divisor
    if div is None:
        div = sum(val for row in kernel for val in row)
        if abs(div) < 1e-6:
            div = 1.0
    out = []
    for r in range(height):
        out_row = []
        for c in range(width):
            acc_r = acc_g = acc_b = 0.0
            for ky in range(k_h):
                src_y = max(0, min(height - 1, r + ky - pad_y))
                for kx in range(k_w):
                    src_x = max(0, min(width - 1, c + kx - pad_x))
                    w = kernel[ky][kx]
                    px = image[src_y][src_x]
                    acc_r += px[0] * w
                    acc_g += px[1] * w
                    acc_b += px[2] * w
            res_r = clamp_rgb(acc_r / div + offset)
            res_g = clamp_rgb(acc_g / div + offset)
            res_b = clamp_rgb(acc_b / div + offset)
            out_row.append((res_r, res_g, res_b))
        out.append(out_row)
    return out

def gaussian_blur(image: Image, radius: int = 1, sigma: float = 1.0) -> Image:
    if radius <= 0:
        return [list(r) for r in image]
    kernel = []
    s2 = 2.0 * sigma * sigma
    for y in range(-radius, radius + 1):
        row = []
        for x in range(-radius, radius + 1):
            v = math.exp(-(x * x + y * y) / s2)
            row.append(v)
        kernel.append(row)
    return convolve_2d(image, kernel)

def box_blur(image: Image, radius: int = 1) -> Image:
    if radius <= 0:
        return [list(r) for r in image]
    size = 2 * radius + 1
    kernel = [[1.0 for _ in range(size)] for _ in range(size)]
    return convolve_2d(image, kernel, divisor=float(size * size))

def sharpen(image: Image, strength: float = 1.0) -> Image:
    s = max(0.0, float(strength))
    if s == 0.0:
        return [list(r) for r in image]
    kernel = [
        [0.0, -s, 0.0],
        [-s, 1.0 + 4.0 * s, -s],
        [0.0, -s, 0.0],
    ]
    return convolve_2d(image, kernel, divisor=1.0)

def sobel_edges(image: Image) -> Image:
    height, width = shape(image)
    def luma(px):
        return (2126 * px[0] + 7152 * px[1] + 722 * px[2]) / 10000.0
    out = []
    for r in range(height):
        row = []
        for c in range(width):
            tl = luma(image[max(0, r - 1)][max(0, c - 1)])
            tc = luma(image[max(0, r - 1)][c])
            tr = luma(image[max(0, r - 1)][min(width - 1, c + 1)])
            ml = luma(image[r][max(0, c - 1)])
            mr = luma(image[r][min(width - 1, c + 1)])
            bl = luma(image[min(height - 1, r + 1)][max(0, c - 1)])
            bc = luma(image[min(height - 1, r + 1)][c])
            br = luma(image[min(height - 1, r + 1)][min(width - 1, c + 1)])
            gx = (tr + 2.0 * mr + br) - (tl + 2.0 * ml + bl)
            gy = (bl + 2.0 * bc + br) - (tl + 2.0 * tc + tr)
            mag = clamp_rgb(math.sqrt(gx * gx + gy * gy) / 4.0)
            row.append((mag, mag, mag))
        out.append(row)
    return out

def emboss(image: Image) -> Image:
    kernel = [
        [-2.0, -1.0, 0.0],
        [-1.0,  1.0, 1.0],
        [ 0.0,  1.0, 2.0],
    ]
    return convolve_2d(image, kernel, divisor=1.0, offset=128.0)

def laplacian(image: Image) -> Image:
    kernel = [
        [0.0,  1.0, 0.0],
        [1.0, -4.0, 1.0],
        [0.0,  1.0, 0.0],
    ]
    return convolve_2d(image, kernel, divisor=1.0, offset=128.0)
