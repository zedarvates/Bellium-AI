"""Controlled renders with known albedo and illumination.

Evaluation only: every case carries its ground-truth maps, so a separation can be
scored against what generated it. These are not photographs and they establish
nothing about real captures.
"""

from __future__ import annotations

import math
import random

from bellium.knn._image import Image, clamp_rgb

Floats = list[list[float]]
ColorFloats = list[list[tuple[float, float, float]]]
EPSILON = 1.0 / 512.0


def _gray(value: float) -> tuple[float, float, float]:
    level = min(max(value, 0.0), 1.0)
    return level, level, level


def procedural_albedo(kind: str, size: int = 48, seed: int = 1) -> ColorFloats:
    """Deterministic albedo base colour, in linear [0, 1] floats."""
    rng = random.Random(seed)
    if kind == "flat":
        level = 0.35 + 0.3 * rng.random()
        return [[_gray(level) for _ in range(size)] for _ in range(size)]
    if kind == "gradient":
        return [
            [_gray(0.15 + 0.7 * (c / (size - 1))) for c in range(size)]
            for _ in range(size)
        ]
    if kind == "checker":
        cell = max(3, size // 8)
        return [
            [
                _gray(0.75 if ((r // cell) + (c // cell)) % 2 else 0.25)
                for c in range(size)
            ]
            for r in range(size)
        ]
    if kind == "stripes":
        period = max(4, size // 6)
        return [
            [_gray(0.6 + 0.3 * math.sin(2 * math.pi * c / period)) for c in range(size)]
            for _ in range(size)
        ]
    if kind == "noise":
        return [
            [
                (value, value * 0.85, value * 0.7)
                for value in (0.15 + 0.7 * rng.random() for _ in range(size))
            ]
            for _ in range(size)
        ]
    if kind == "color-blocks":
        block = max(4, size // 4)
        palette = [(0.75, 0.25, 0.2), (0.2, 0.5, 0.75), (0.7, 0.7, 0.2), (0.25, 0.6, 0.3)]
        return [
            [
                palette[((r // block) + (c // block)) % len(palette)]
                for c in range(size)
            ]
            for r in range(size)
        ]
    raise ValueError(f"unknown albedo kind: {kind}")


def illumination_field(kind: str, size: int = 48, *, strength: float = 0.7) -> Floats:
    """Deterministic illumination multiplier in linear [0, 1] floats."""
    if isinstance(strength, bool) or not isinstance(strength, (int, float)):
        raise ValueError("strength must be numeric")
    if not 0.0 < float(strength) <= 1.0:
        raise ValueError("strength must be in (0, 1]")
    span = float(strength)
    center = (size - 1) / 2.0
    if kind == "constant":
        return [[1.0 for _ in range(size)] for _ in range(size)]
    if kind == "linear":
        return [
            [1.0 - span * (r / (size - 1)) for _ in range(size)] for r in range(size)
        ]
    if kind == "vignette":
        limit = math.hypot(center, center)
        return [
            [
                1.0 - span * (math.hypot(r - center, c - center) / limit)
                for c in range(size)
            ]
            for r in range(size)
        ]
    if kind == "lamp":
        return [
            [
                min(1.0, 0.25 + span * (1.0 / (1.0 + ((r - center) ** 2 + (c - center) ** 2) / (size * 1.2))))
                for c in range(size)
            ]
            for r in range(size)
        ]
    if kind == "shadow-band":
        limit = max(2, size // 6)
        return [
            [
                0.35 if ((c - size // 3) % (size // 2)) < limit else 1.0
                for c in range(size)
            ]
            for r in range(size)
        ]
    raise ValueError(f"unknown illumination kind: {kind}")


def render(albedo: ColorFloats, illumination: Floats, *, gamma: float = 1.0) -> ColorFloats:
    """Lambertian product of albedo and illumination, optionally gamma encoded."""
    if isinstance(gamma, bool) or not isinstance(gamma, (int, float)):
        raise ValueError("gamma must be numeric")
    if not math.isfinite(float(gamma)) or float(gamma) <= 0.0:
        raise ValueError("gamma must be positive and finite")
    size = len(albedo)
    if len(illumination) != size or any(
        len(row) != len(albedo[0]) for row in illumination
    ):
        raise ValueError("albedo and illumination must share a shape")
    out: ColorFloats = []
    for r in range(size):
        row = []
        for c in range(len(albedo[0])):
            light = min(max(illumination[r][c], 0.0), 1.0)
            pixel = []
            for channel in albedo[r][c]:
                linear = min(max(channel * light, 0.0), 1.0)
                pixel.append(min(max(linear ** (1.0 / float(gamma)), 0.0), 1.0))
            row.append((pixel[0], pixel[1], pixel[2]))
        out.append(row)
    return out


def render_case(albedo_kind: str, illumination_kind: str, *, size: int = 48, seed: int = 1,
                strength: float = 0.7, gamma: float = 1.0) -> dict:
    """One scored case with its ground truth."""
    albedo = procedural_albedo(albedo_kind, size=size, seed=seed)
    illumination = illumination_field(illumination_kind, size=size, strength=strength)
    rendered = render(albedo, illumination, gamma=gamma)
    return {
        "name": f"{albedo_kind}+{illumination_kind}",
        "albedo": albedo,
        "illumination": illumination,
        "render": rendered,
        "albedo_kind": albedo_kind,
        "illumination_kind": illumination_kind,
        "strength": strength,
        "gamma": gamma,
        "seed": seed,
        "size": size,
    }


def to_image(source: ColorFloats) -> Image:
    """Convert a float render to the repository RGB integer contract."""
    return [
        [(clamp_rgb(pixel[0] * 255.0), clamp_rgb(pixel[1] * 255.0), clamp_rgb(pixel[2] * 255.0))
         for pixel in row]
        for row in source
    ]


def from_image(image: Image) -> ColorFloats:
    """Convert an integer RGB image to linear [0, 1] floats."""
    return [
        [(pixel[0] / 255.0, pixel[1] / 255.0, pixel[2] / 255.0) for pixel in row]
        for row in image
    ]


def to_gray(source: ColorFloats) -> Floats:
    return [
        [
            (0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.0722 * pixel[2])
            for pixel in row
        ]
        for row in source
    ]
