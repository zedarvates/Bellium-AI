from __future__ import annotations

import math
from bellium.editing.resample import resample_bilinear
from bellium.knn._image import Image, Rgb, shape

def create_montage(
    images: list[Image],
    *,
    columns: int | None = None,
    rows: int | None = None,
    tile_size: tuple[int, int] | None = None,
    padding: int = 2,
    background: Rgb = (20, 20, 20),
) -> Image:
    if not images:
        raise ValueError('montage requires at least one image')
    for img in images:
        shape(img)
    n = len(images)
    if columns is None and rows is None:
        columns = math.ceil(math.sqrt(n))
        rows = math.ceil(n / columns)
    elif columns is None:
        assert rows is not None and rows > 0
        columns = math.ceil(n / rows)
    elif rows is None:
        assert columns > 0
        rows = math.ceil(n / columns)
    if tile_size is None:
        tile_h = max(len(img) for img in images)
        tile_w = max(len(img[0]) for img in images)
    else:
        tile_w, tile_h = tile_size
        if tile_w <= 0 or tile_h <= 0:
            raise ValueError('tile_size dimensions must be positive')
    pad = max(0, int(padding))
    total_w = columns * tile_w + (columns + 1) * pad
    total_h = rows * tile_h + (rows + 1) * pad
    canvas = [[background for _ in range(total_w)] for _ in range(total_h)]
    for idx, img in enumerate(images):
        if idx >= columns * rows:
            break
        grid_c = idx % columns
        grid_r = idx // columns
        ih, iw = len(img), len(img[0])
        if ih != tile_h or iw != tile_w:
            scaled = resample_bilinear(img, tile_h, tile_w)
        else:
            scaled = img
        y_start = pad + grid_r * (tile_h + pad)
        x_start = pad + grid_c * (tile_w + pad)
        for y in range(tile_h):
            for x in range(tile_w):
                canvas[y_start + y][x_start + x] = scaled[y][x]
    return canvas
