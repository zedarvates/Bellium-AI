from __future__ import annotations

from bellium.knn._image import Image, Rgb, shape

def flip_horizontal(image: Image) -> Image:
    shape(image)
    return [list(reversed(row)) for row in image]

def flip_vertical(image: Image) -> Image:
    shape(image)
    return [list(row) for row in reversed(image)]

def rotate_90_cw(image: Image) -> Image:
    height, width = shape(image)
    out = []
    for c in range(width):
        row = [image[r][c] for r in range(height - 1, -1, -1)]
        out.append(row)
    return out

def rotate_180(image: Image) -> Image:
    return flip_vertical(flip_horizontal(image))

def rotate_270_cw(image: Image) -> Image:
    height, width = shape(image)
    out = []
    for c in range(width - 1, -1, -1):
        row = [image[r][c] for r in range(height)]
        out.append(row)
    return out

def pad_extent(
    image: Image,
    target_height: int,
    target_width: int,
    *,
    background: Rgb = (0, 0, 0),
    anchor: str = 'center',
) -> Image:
    in_h, in_w = shape(image)
    if target_height < in_h or target_width < in_w:
        raise ValueError('extent target dimensions must be at least source dimensions')
    if anchor == 'center':
        offset_y = (target_height - in_h) // 2
        offset_x = (target_width - in_w) // 2
    elif anchor == 'top_left':
        offset_y = offset_x = 0
    else:
        raise ValueError(f'unsupported anchor: {anchor}')
    out = [[background for _ in range(target_width)] for _ in range(target_height)]
    for r in range(in_h):
        for c in range(in_w):
            out[r + offset_y][c + offset_x] = image[r][c]
    return out

def trim_uniform(image: Image, tolerance: float = 0.0) -> tuple[Image, tuple[int, int, int, int]]:
    height, width = shape(image)
    bg = image[0][0]
    def match(px):
        diff = abs(px[0] - bg[0]) + abs(px[1] - bg[1]) + abs(px[2] - bg[2])
        return diff <= tolerance * 3.0
    top = 0
    while top < height and all(match(image[top][c]) for c in range(width)):
        top += 1
    if top == height:
        return [[bg]], (0, 0, 1, 1)
    bottom = height - 1
    while bottom > top and all(match(image[bottom][c]) for c in range(width)):
        bottom -= 1
    left = 0
    while left < width and all(match(image[r][left]) for r in range(top, bottom + 1)):
        left += 1
    right = width - 1
    while right > left and all(match(image[r][right]) for r in range(top, bottom + 1)):
        right -= 1
    cropped = [image[r][left : right + 1] for r in range(top, bottom + 1)]
    return cropped, (top, left, bottom + 1, right + 1)
