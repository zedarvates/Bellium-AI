from __future__ import annotations

import math
from bellium.knn._image import Image, Rgb, shape

def slice_spritesheet(
    sheet: Image,
    frame_width: int,
    frame_height: int,
) -> list[Image]:
    h, w = shape(sheet)
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError('frame dimensions must be positive integers')
    if w < frame_width or h < frame_height:
        raise ValueError(f'sheet {w}x{h} is smaller than frame {frame_width}x{frame_height}')
    cols = w // frame_width
    rows = h // frame_height
    frames: list[Image] = []
    for r in range(rows):
        for c in range(cols):
            y_start = r * frame_height
            x_start = c * frame_width
            frame = [
                sheet[y_start + y][x_start : x_start + frame_width]
                for y in range(frame_height)
            ]
            frames.append(frame)
    return frames

def assemble_spritesheet(
    frames: list[Image],
    *,
    direction: str = 'horizontal',
    columns: int | None = None,
    background: Rgb = (0, 0, 0),
) -> Image:
    if not frames:
        raise ValueError('assemble_spritesheet requires at least one frame')
    for f in frames:
        shape(f)
    f_h, f_w = len(frames[0]), len(frames[0][0])
    for f in frames:
        if len(f) != f_h or len(f[0]) != f_w:
            raise ValueError('all frames must have identical dimensions')
    n = len(frames)
    if direction == 'horizontal':
        out = []
        for y in range(f_h):
            row = []
            for f in frames:
                row.extend(f[y])
            out.append(row)
        return out
    elif direction == 'vertical':
        out = []
        for f in frames:
            for y in range(f_h):
                out.append(list(f[y]))
        return out
    elif direction == 'grid':
        cols = columns if (columns and columns > 0) else math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
        total_w = cols * f_w
        total_h = rows * f_h
        canvas = [[background for _ in range(total_w)] for _ in range(total_h)]
        for idx, f in enumerate(frames):
            grid_r = idx // cols
            grid_c = idx % cols
            y_offset = grid_r * f_h
            x_offset = grid_c * f_w
            for y in range(f_h):
                for x in range(f_w):
                    canvas[y_offset + y][x_offset + x] = f[y][x]
        return canvas
    else:
        raise ValueError(f'unsupported direction: {direction}; expected horizontal, vertical, or grid')
