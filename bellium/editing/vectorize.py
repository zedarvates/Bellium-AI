from __future__ import annotations

import math
from bellium.editing.quantize import extract_palette_median_cut, quantize_nearest
from bellium.knn._image import Image, Rgb, shape

Point = tuple[float, float]


def _add_edge(fwd: dict[Point, list[Point]], a: Point, b: Point) -> None:
    reverse = fwd.get(b)
    if reverse and a in reverse:
        reverse.remove(a)
        if not reverse:
            del fwd[b]
        return
    fwd.setdefault(a, []).append(b)


def _region_cycles(pixels: list[tuple[int, int]]) -> list[list[Point]]:
    fwd: dict[Point, list[Point]] = {}
    for row, col in pixels:
        x = float(col)
        y = float(row)
        _add_edge(fwd, (x, y), (x + 1.0, y))
        _add_edge(fwd, (x + 1.0, y), (x + 1.0, y + 1.0))
        _add_edge(fwd, (x + 1.0, y + 1.0), (x, y + 1.0))
        _add_edge(fwd, (x, y + 1.0), (x, y))
    cycles: list[list[Point]] = []
    while fwd:
        start = next(iter(fwd))
        cycle = [start]
        current = start
        while current in fwd and fwd[current]:
            nxt = fwd[current].pop(0)
            if not fwd[current]:
                del fwd[current]
            if nxt == start:
                break
            cycle.append(nxt)
            current = nxt
            if len(cycle) > 100000:
                break
        if len(cycle) >= 3:
            cycles.append(cycle)
    cycles.sort(key=lambda pts: abs(_polygon_area(pts)), reverse=True)
    return cycles


def _polygon_area(points: list[Point]) -> float:
    total = 0.0
    count = len(points)
    for i, (x1, y1) in enumerate(points):
        x2, y2 = points[(i + 1) % count]
        total += x1 * y2 - x2 * y1
    return 0.5 * total


def _perp_dist(point: Point, start: Point, end: Point) -> float:
    x, y = point
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)
    if length < 1e-9:
        return math.hypot(x - x1, y - y1)
    return abs(dy * x - dx * y + x2 * y1 - y2 * x1) / length


def simplify_path(points: list[Point], epsilon: float) -> list[Point]:
    if len(points) < 3 or epsilon <= 0.0:
        return list(points)
    max_dist = -1.0
    index = 0
    start = points[0]
    end = points[-1]
    for i in range(1, len(points) - 1):
        dist = _perp_dist(points[i], start, end)
        if dist > max_dist:
            index = i
            max_dist = dist
    if max_dist > epsilon:
        left = simplify_path(points[: index + 1], epsilon)
        right = simplify_path(points[index:], epsilon)
        return left[:-1] + right
    return [start, end]


def _path_d(points: list[Point]) -> str:
    if not points:
        return ""
    parts = [f"M {points[0][0]:.2f} {points[0][1]:.2f}"]
    for x, y in points[1:]:
        parts.append(f"L {x:.2f} {y:.2f}")
    parts.append("Z")
    return " ".join(parts)


def _connected_components(image: Image) -> list[tuple[Rgb, list[tuple[int, int]]]]:
    height, width = shape(image)
    seen = [[False] * width for _ in range(height)]
    components: list[tuple[Rgb, list[tuple[int, int]]]] = []
    for r in range(height):
        for c in range(width):
            if seen[r][c]:
                continue
            color = image[r][c]
            stack = [(r, c)]
            seen[r][c] = True
            pixels = []
            while stack:
                y, x = stack.pop()
                pixels.append((y, x))
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= ny < height and 0 <= nx < width and not seen[ny][nx] and image[ny][nx] == color:
                        seen[ny][nx] = True
                        stack.append((ny, nx))
            components.append((color, pixels))
    components.sort(key=lambda item: len(item[1]), reverse=True)
    return components


def region_features(pixels: list[tuple[int, int]], color: Rgb, image: Image) -> dict[str, float]:
    area = len(pixels)
    if area == 0:
        raise ValueError("region is empty")
    rows = [p[0] for p in pixels]
    cols = [p[1] for p in pixels]
    height, width = shape(image)
    pixel_set = set(pixels)
    perimeter = 0
    for y, x in pixels:
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if (ny, nx) not in pixel_set:
                perimeter += 1
    compactness = 0.0 if perimeter == 0 else min(1.0, (4.0 * math.pi * area) / (perimeter * perimeter))
    bbox_h = max(rows) - min(rows) + 1
    bbox_w = max(cols) - min(cols) + 1
    fill_ratio = min(1.0, area / max(1, bbox_h * bbox_w))
    lumas = []
    quantized = set()
    for y, x in pixels:
        px = image[y][x]
        lumas.append((2126 * px[0] + 7152 * px[1] + 722 * px[2]) / 10000.0)
        quantized.add((px[0] // 16, px[1] // 16, px[2] // 16))
    mean_l = sum(lumas) / len(lumas)
    var_l = sum((v - mean_l) ** 2 for v in lumas) / len(lumas)
    chroma = (max(color) - min(color)) / 255.0
    return {
        "luma_std": min(1.0, math.sqrt(var_l) / 128.0),
        "unique_ratio": min(1.0, len(quantized) / area),
        "compactness": compactness,
        "fill_ratio": fill_ratio,
        "chroma": min(1.0, chroma),
    }


def vectorize_regions(
    image: Image,
    *,
    max_colors: int = 8,
    min_area: int = 4,
    epsilon: float = 0.6,
    skip_background: bool = True,
    max_regions: int = 64,
) -> dict[str, object]:
    height, width = shape(image)
    if isinstance(max_colors, bool) or not isinstance(max_colors, int) or max_colors <= 0:
        raise ValueError("max_colors must be a positive integer")
    palette = extract_palette_median_cut(image, max_colors=max_colors)
    quantized = quantize_nearest(image, palette)
    components = _connected_components(quantized)
    background = components[0][0] if components else (255, 255, 255)
    regions = []
    for color, pixels in components:
        if skip_background and color == background and len(pixels) > (height * width) // 4:
            continue
        if len(pixels) < min_area:
            continue
        cycles = _region_cycles(pixels)
        if not cycles:
            continue
        simplified = [simplify_path(cycle + [cycle[0]], epsilon)[:-1] for cycle in cycles]
        simplified = [pts for pts in simplified if len(pts) >= 3]
        if not simplified:
            continue
        features = region_features(pixels, color, quantized)
        regions.append({
            "color": color,
            "area": len(pixels),
            "paths": simplified,
            "features": features,
        })
        if len(regions) >= max_regions:
            break
    return {
        "width": width,
        "height": height,
        "background": background,
        "palette": palette,
        "quantized": quantized,
        "regions": regions,
    }


def regions_to_svg(payload: dict[str, object]) -> str:
    width = int(payload["width"])
    height = int(payload["height"])
    background = payload["background"]
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" shape-rendering="crispEdges">',
        f'  <rect width="100%" height="100%" fill="rgb({background[0]},{background[1]},{background[2]})" />',
    ]
    for region in reversed(payload["regions"]):
        color = region["color"]
        fill = f"rgb({color[0]},{color[1]},{color[2]})"
        for points in region["paths"]:
            lines.append(f'  <path d="{_path_d(points)}" fill="{fill}" stroke="none" />')
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _point_in_poly(x: float, y: float, points: list[Point]) -> bool:
    inside = False
    j = len(points) - 1
    for i, (xi, yi) in enumerate(points):
        xj, yj = points[j]
        intersect = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi)
        if intersect:
            inside = not inside
        j = i
    return inside


def rasterize_regions(payload: dict[str, object]) -> Image:
    width = int(payload["width"])
    height = int(payload["height"])
    background = payload["background"]
    out = [[background for _ in range(width)] for _ in range(height)]
    for region in reversed(payload["regions"]):
        color = region["color"]
        for points in region["paths"]:
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            x0 = max(0, int(math.floor(min(xs))))
            x1 = min(width - 1, int(math.ceil(max(xs))))
            y0 = max(0, int(math.floor(min(ys))))
            y1 = min(height - 1, int(math.ceil(max(ys))))
            for y in range(y0, y1 + 1):
                cy = y + 0.5
                for x in range(x0, x1 + 1):
                    if _point_in_poly(x + 0.5, cy, points):
                        out[y][x] = color
    return out


def fill_disc(image: Image, cx: int, cy: int, radius: int, color: Rgb) -> None:
    height, width = shape(image)
    r2 = radius * radius
    for y in range(max(0, cy - radius), min(height, cy + radius + 1)):
        for x in range(max(0, cx - radius), min(width, cx + radius + 1)):
            if (x - cx) * (x - cx) + (y - cy) * (y - cy) <= r2:
                image[y][x] = color


def fill_rect(image: Image, x0: int, y0: int, x1: int, y1: int, color: Rgb) -> None:
    height, width = shape(image)
    xa, xb = sorted((x0, x1))
    ya, yb = sorted((y0, y1))
    for y in range(max(0, ya), min(height, yb + 1)):
        for x in range(max(0, xa), min(width, xb + 1)):
            image[y][x] = color


def make_pelican_bicycle(width: int = 64, height: int = 48) -> Image:
    """Original geometric scene: a flat-color pelican on a bicycle.

    This is a vectorization fixture, not a generative-model exam and not a copy of
    any published illustration. YouTubers comparing image generators still need
    a real prompt-to-pixels test; this drawing only checks that hard-edge art
    becomes SVG paths.
    """
    if width < 32 or height < 24:
        raise ValueError("pelican fixture needs at least 32x24")
    sky = (236, 246, 255)
    ink = (28, 28, 32)
    body = (248, 248, 242)
    wing = (220, 224, 214)
    beak = (236, 132, 36)
    feet = (244, 196, 48)
    image = [[sky for _ in range(width)] for _ in range(height)]
    fill_disc(image, 18, 36, 8, ink)
    fill_disc(image, 18, 36, 4, sky)
    fill_disc(image, 46, 36, 8, ink)
    fill_disc(image, 46, 36, 4, sky)
    fill_rect(image, 18, 35, 46, 36, ink)
    fill_rect(image, 32, 22, 33, 35, ink)
    fill_rect(image, 22, 24, 32, 25, ink)
    fill_rect(image, 33, 24, 44, 25, ink)
    fill_rect(image, 26, 18, 38, 28, body)
    fill_rect(image, 22, 20, 27, 26, wing)
    fill_rect(image, 38, 20, 48, 23, beak)
    fill_rect(image, 48, 21, 52, 22, beak)
    fill_rect(image, 34, 20, 35, 21, ink)
    fill_rect(image, 30, 28, 31, 35, feet)
    fill_rect(image, 34, 28, 35, 35, feet)
    return image
