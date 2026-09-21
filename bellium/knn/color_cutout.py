"""Border-color k-NN cutout for hard-edge objects on simple backgrounds."""

from __future__ import annotations

import math
from collections import Counter
from heapq import nsmallest

from bellium.contracts.schema import AuthorityMode, SpecialistResult
from bellium.knn._image import Image, Rgb, clamp_rgb, copy_image, shape

SPECIALIST_ID = "bellium/knn/color-cutout:v0"
BORDER = 2
K = 5
DISTANCE_THRESHOLD = 48.0
MAX_BORDER_COLORS = 512
MAX_CACHED_COLORS = 4096


def _border_samples(image: Image, band: int) -> list[Rgb]:
    height, width = shape(image)
    samples: list[Rgb] = []
    for r in range(height):
        for c in range(width):
            if r < band or c < band or r >= height - band or c >= width - band:
                samples.append(image[r][c])
    return samples


def _color_std(samples: list[Rgb]) -> float:
    if not samples:
        return 1.0
    means = [sum(pixel[i] for pixel in samples) / len(samples) for i in range(3)]
    var = sum((pixel[i] - means[i]) ** 2 for pixel in samples for i in range(3)) / (len(samples) * 3)
    return math.sqrt(var) / 255.0


def _distance(pixel: Rgb, samples: Counter) -> float:
    # Preserve the original vote multiplicities without comparing every repeated
    # border pixel or sorting the complete border for every image pixel.
    nearest = nsmallest(K, (
        (sum((pixel[i] - other[i]) ** 2 for i in range(3)), count)
        for other, count in samples.items()
    ))
    total = 0.0
    used = 0
    for distance, count in nearest:
        take = min(count, K - used)
        total += math.sqrt(distance) * take
        used += take
        if used == K:
            break
    return total / used


def cutout(image: Image, *, threshold: float = DISTANCE_THRESHOLD) -> SpecialistResult:
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be positive and finite")
    height, width = shape(image)
    samples = _border_samples(image, BORDER)
    spread = _color_std(samples)
    if spread > 0.22:
        return SpecialistResult(
            SPECIALIST_ID,
            {"reason": "busy_border", "border_std": round(spread, 4)},
            0.0, True, AuthorityMode.CONSULTATIVE,
            notes=("Background is too mixed for a color k-NN cutout.",),
        )
    color_counts = Counter(tuple(pixel) for pixel in samples)
    if len(color_counts) > MAX_BORDER_COLORS:
        return SpecialistResult(
            SPECIALIST_ID, {"reason": "too_many_border_colors"},
            0.0, True, AuthorityMode.CONSULTATIVE,
            notes=("Exact color search exceeds the bounded border budget.",),
        )
    cache = {}
    alpha: list[list[float]] = []
    fg = 0
    distances: list[float] = []
    for r in range(height):
        row: list[float] = []
        for c in range(width):
            pixel_key = tuple(image[r][c])
            dist = cache.get(pixel_key)
            if dist is None:
                dist = _distance(pixel_key, color_counts)
                if len(cache) < MAX_CACHED_COLORS:
                    cache[pixel_key] = dist
            distances.append(dist)
            value = 1.0 if dist >= threshold else 0.0
            if value:
                fg += 1
            row.append(value)
        alpha.append(row)
    ratio = fg / (height * width)
    if ratio < 0.02 or ratio > 0.92:
        return SpecialistResult(
            SPECIALIST_ID,
            {"reason": "implausible_foreground", "foreground_ratio": round(ratio, 4)},
            0.0, True, AuthorityMode.CONSULTATIVE,
            notes=("Cutout looks empty or inverted; abstaining.",),
        )
    rgba = copy_image(image)
    for r in range(height):
        for c in range(width):
            pixel = image[r][c]
            a = 255 if alpha[r][c] else 0
            rgba[r][c] = (pixel[0], pixel[1], pixel[2], a)  # type: ignore[assignment]
    margin = (sum(distances) / len(distances) - threshold) / 255.0
    confidence = max(0.0, min(0.95, 0.55 + margin + (0.22 - spread)))
    return SpecialistResult(
        SPECIALIST_ID,
        {
            "rgba": rgba,
            "alpha": alpha,
            "foreground_ratio": round(ratio, 4),
            "border_std": round(spread, 4),
        },
        round(confidence, 4),
        False,
        AuthorityMode.CONSULTATIVE,
        notes=("Hard-edge color k-NN only. Hair, glass and busy scenes must escalate.",),
    )


def to_white_background(rgba_image: list[list[tuple[int, int, int, int]]]) -> Image:
    out: Image = []
    for row in rgba_image:
        painted = []
        for r, g, b, a in row:
            alpha = clamp_rgb(a) / 255.0
            painted.append(tuple(clamp_rgb(channel * alpha + 255 * (1 - alpha)) for channel in (r, g, b)))
        out.append(painted)
    return out
