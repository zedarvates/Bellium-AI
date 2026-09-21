"""Deterministic selection masks: shapes, brush strokes, feathering and blending.

Masks are float maps in [0, 1]. A masked edit keeps every pixel the mask does not
cover exactly as it was, which is the invariant the editing gate checks.
"""

from __future__ import annotations

import math

from bellium.knn._image import Image, clamp_rgb, copy_image, shape

Mask = list[list[float]]


def _size(size: object) -> tuple[int, int]:
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        raise ValueError("size must list width and height")
    width, height = size
    for value in (width, height):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("size values must be positive integers")
    return width, height


def _empty(width: int, height: int) -> Mask:
    return [[0.0] * width for _ in range(height)]


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def rect_mask(size: object, box: object) -> Mask:
    width, height = _size(size)
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise ValueError("box must list x, y, w and h")
    x, y, w, h = (_number(value, "box") for value in box)
    mask = _empty(width, height)
    for r in range(height):
        for c in range(width):
            if x <= c < x + w and y <= r < y + h:
                mask[r][c] = 1.0
    return mask


def ellipse_mask(size: object, center: object, radii: object) -> Mask:
    width, height = _size(size)
    if not isinstance(center, (list, tuple)) or len(center) != 2:
        raise ValueError("center must list x and y")
    if not isinstance(radii, (list, tuple)) or len(radii) != 2:
        raise ValueError("radii must list two values")
    cx, cy = (_number(value, "center") for value in center)
    rx, ry = (_number(value, "radii") for value in radii)
    if rx <= 0.0 or ry <= 0.0:
        raise ValueError("radii must be positive")
    mask = _empty(width, height)
    for r in range(height):
        for c in range(width):
            if ((c - cx) / rx) ** 2 + ((r - cy) / ry) ** 2 <= 1.0:
                mask[r][c] = 1.0
    return mask


def polygon_mask(size: object, points: object) -> Mask:
    width, height = _size(size)
    if not isinstance(points, (list, tuple)) or len(points) < 3:
        raise ValueError("a polygon needs at least three points")
    vertices = []
    for index, point in enumerate(points):
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError(f"point {index} must list x and y")
        vertices.append(
            (_number(point[0], f"point {index} x"), _number(point[1], f"point {index} y"))
        )
    mask = _empty(width, height)
    for r in range(height):
        for c in range(width):
            inside = False
            for index in range(len(vertices)):
                x1, y1 = vertices[index]
                x2, y2 = vertices[(index + 1) % len(vertices)]
                if (y1 > r) != (y2 > r):
                    crossing = (x2 - x1) * (r - y1) / (y2 - y1) + x1
                    if c < crossing:
                        inside = not inside
            mask[r][c] = 1.0 if inside else 0.0
    return mask


def brush_mask(size: object, path: object, *, radius: object = 4.0) -> Mask:
    """Discs stamped along a path, so a stroke covers the segments between points."""
    width, height = _size(size)
    if not isinstance(path, (list, tuple)) or not path:
        raise ValueError("path must list at least one point")
    points = []
    for index, point in enumerate(path):
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError(f"point {index} must list x and y")
        points.append((_number(point[0], "x"), _number(point[1], "y")))
    reach = _number(radius, "radius")
    if reach <= 0.0:
        raise ValueError("radius must be positive")
    mask = _empty(width, height)
    for index in range(len(points)):
        x1, y1 = points[index]
        x2, y2 = points[index + 1] if index + 1 < len(points) else points[index]
        steps = max(1, int(math.ceil(math.hypot(x2 - x1, y2 - y1))))
        for step in range(steps + 1):
            fraction = step / steps
            cx = x1 + (x2 - x1) * fraction
            cy = y1 + (y2 - y1) * fraction
            for r in range(max(0, int(cy - reach)), min(height, int(cy + reach) + 1)):
                for c in range(max(0, int(cx - reach)), min(width, int(cx + reach) + 1)):
                    if (c - cx) ** 2 + (r - cy) ** 2 <= reach * reach:
                        mask[r][c] = 1.0
    return mask


def feather(mask: object, *, radius: object = 2) -> Mask:
    """Box blur the mask so an edit fades instead of ending on a hard line."""
    if not isinstance(mask, list) or not mask or not mask[0]:
        raise ValueError("mask is empty")
    if isinstance(radius, bool) or not isinstance(radius, int) or radius < 1:
        raise ValueError("radius must be an integer of at least one")
    height = len(mask)
    width = len(mask[0])
    for row in mask:
        if len(row) != width:
            raise ValueError("mask rows must have equal width")
        for value in row:
            _unit_value(value)
    horizontal = [[0.0] * width for _ in range(height)]
    for r in range(height):
        for c in range(width):
            total = 0.0
            for offset in range(-radius, radius + 1):
                total += mask[r][min(max(c + offset, 0), width - 1)]
            horizontal[r][c] = total / (2 * radius + 1)
    out = [[0.0] * width for _ in range(height)]
    for c in range(width):
        for r in range(height):
            total = 0.0
            for offset in range(-radius, radius + 1):
                total += horizontal[min(max(r + offset, 0), height - 1)][c]
            out[r][c] = total / (2 * radius + 1)
    return out


def _unit_value(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("mask values must be numeric")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError("mask values must be between 0 and 1")
    return number


def apply_mask(image: Image, edited: Image, mask: object) -> Image:
    """Blend edited over image by the mask; uncovered pixels stay byte for byte."""
    height, width = shape(image)
    if len(edited) != height or any(len(row) != width for row in edited):
        raise ValueError("the edited image must share the source shape")
    if not isinstance(mask, list) or len(mask) != height or any(
        len(row) != width for row in mask
    ):
        raise ValueError("the mask must share the image shape")
    out = copy_image(image)
    for r in range(height):
        for c in range(width):
            weight = _unit_value(mask[r][c])
            if weight <= 0.0:
                continue
            source = image[r][c]
            target = edited[r][c]
            out[r][c] = (
                clamp_rgb(source[0] + (target[0] - source[0]) * weight),
                clamp_rgb(source[1] + (target[1] - source[1]) * weight),
                clamp_rgb(source[2] + (target[2] - source[2]) * weight),
            )
    return out
