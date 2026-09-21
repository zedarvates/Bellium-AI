from __future__ import annotations

from bellium.knn._image import Image, clamp_rgb, shape

def _validate_target_dims(target_height: int, target_width: int) -> None:
    if isinstance(target_height, bool) or not isinstance(target_height, int) or target_height <= 0:
        raise ValueError('target_height must be a positive integer')
    if isinstance(target_width, bool) or not isinstance(target_width, int) or target_width <= 0:
        raise ValueError('target_width must be a positive integer')

def resample_nearest(image: Image, target_height: int, target_width: int) -> Image:
    in_h, in_w = shape(image)
    _validate_target_dims(target_height, target_width)
    out = []
    for r in range(target_height):
        src_r = min(int(r * in_h / target_height), in_h - 1)
        row = []
        for c in range(target_width):
            src_c = min(int(c * in_w / target_width), in_w - 1)
            row.append(image[src_r][src_c])
        out.append(row)
    return out

def resample_bilinear(image: Image, target_height: int, target_width: int) -> Image:
    in_h, in_w = shape(image)
    _validate_target_dims(target_height, target_width)
    out = []
    for r in range(target_height):
        src_y = (r + 0.5) * in_h / target_height - 0.5
        y0 = max(0, min(in_h - 1, int(src_y)))
        y1 = max(0, min(in_h - 1, y0 + 1))
        dy = src_y - y0
        row = []
        for c in range(target_width):
            src_x = (c + 0.5) * in_w / target_width - 0.5
            x0 = max(0, min(in_w - 1, int(src_x)))
            x1 = max(0, min(in_w - 1, x0 + 1))
            dx = src_x - x0
            tl = image[y0][x0]
            tr = image[y0][x1]
            bl = image[y1][x0]
            br = image[y1][x1]
            channels = []
            for ch in range(3):
                top = tl[ch] * (1.0 - dx) + tr[ch] * dx
                bot = bl[ch] * (1.0 - dx) + br[ch] * dx
                val = top * (1.0 - dy) + bot * dy
                channels.append(clamp_rgb(val))
            row.append((channels[0], channels[1], channels[2]))
        out.append(row)
    return out

def resample_box(image: Image, target_height: int, target_width: int) -> Image:
    in_h, in_w = shape(image)
    _validate_target_dims(target_height, target_width)
    out = []
    for r in range(target_height):
        r_start = int(r * in_h / target_height)
        r_end = max(r_start + 1, int((r + 1) * in_h / target_height))
        row = []
        for c in range(target_width):
            c_start = int(c * in_w / target_width)
            c_end = max(c_start + 1, int((c + 1) * in_w / target_width))
            sum_r = sum_g = sum_b = 0
            count = 0
            for y in range(r_start, min(r_end, in_h)):
                for x in range(c_start, min(c_end, in_w)):
                    px = image[y][x]
                    sum_r += px[0]
                    sum_g += px[1]
                    sum_b += px[2]
                    count += 1
            if count == 0:
                row.append(image[min(r_start, in_h - 1)][min(c_start, in_w - 1)])
            else:
                row.append((clamp_rgb(sum_r / count), clamp_rgb(sum_g / count), clamp_rgb(sum_b / count)))
        out.append(row)
    return out
