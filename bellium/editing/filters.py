"""Deterministic direct filters on the repository RGB image contract.

Every filter returns a new image: the source is never written to. Strength blends
between the source and the fully filtered pixel, so a filter can be applied
partially without a second implementation. Colours are handled in the source
space with documented clipping; this is not a colour-managed pipeline.
"""

from __future__ import annotations

import math

from bellium.knn._image import Image, Rgb, clamp_rgb, copy_image, shape

# Integer-scaled coefficients, so the plain and vectorised paths compute the same
# numerator exactly and cannot disagree on a rounding boundary.
LUMA_SCALED = (2126, 7152, 722)
LUMA_DIVISOR = 10000
SEPIA_SCALED = (
    (393, 769, 189),
    (349, 686, 168),
    (272, 534, 131),
)
SEPIA_DIVISOR = 1000


def _unit(value: object, name: str, *, low: float = 0.0, high: float = 1.0) -> float:
    if value is None:
        raise ValueError(f"{name} is unknown; do not coerce to 0")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or not low <= number <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return number


def _luma(pixel: Rgb) -> float:
    return (
        LUMA_SCALED[0] * pixel[0] + LUMA_SCALED[1] * pixel[1] + LUMA_SCALED[2] * pixel[2]
    ) / LUMA_DIVISOR


def _blend(source: Rgb, target: Rgb, strength: float) -> Rgb:
    if strength >= 1.0:
        return target
    if strength <= 0.0:
        return source
    return (
        clamp_rgb(source[0] + (target[0] - source[0]) * strength),
        clamp_rgb(source[1] + (target[1] - source[1]) * strength),
        clamp_rgb(source[2] + (target[2] - source[2]) * strength),
    )


def _map(image: Image, function, strength: float) -> Image:
    shape(image)
    amount = _unit(strength, "strength")
    edited = copy_image(image)
    for r, row in enumerate(image):
        for c, pixel in enumerate(row):
            edited[r][c] = _blend(pixel, function(pixel), amount)
    return edited


def grayscale(image: Image, *, strength: object = 1.0, weights: str = "luma") -> Image:
    """Desaturate towards a grey level, by the documented luma weights or a mean."""
    if weights not in ("luma", "average"):
        raise ValueError("weights must be luma or average")

    def convert(pixel: Rgb) -> Rgb:
        if weights == "luma":
            level = _luma(pixel)
        else:
            level = (pixel[0] + pixel[1] + pixel[2]) / 3.0
        channel = clamp_rgb(level)
        return (channel, channel, channel)

    return _map(image, convert, strength)


def sepia(image: Image, *, strength: object = 1.0) -> Image:
    """Apply the published sepia matrix. A tone, not a colour-managed conversion."""

    def convert(pixel: Rgb) -> Rgb:
        return (
            clamp_rgb(sum(SEPIA_SCALED[0][i] * pixel[i] for i in range(3)) / SEPIA_DIVISOR),
            clamp_rgb(sum(SEPIA_SCALED[1][i] * pixel[i] for i in range(3)) / SEPIA_DIVISOR),
            clamp_rgb(sum(SEPIA_SCALED[2][i] * pixel[i] for i in range(3)) / SEPIA_DIVISOR),
        )

    return _map(image, convert, strength)


def invert(image: Image, *, strength: object = 1.0) -> Image:
    """Invert every channel."""

    def convert(pixel: Rgb) -> Rgb:
        return (255 - pixel[0], 255 - pixel[1], 255 - pixel[2])

    return _map(image, convert, strength)


def brightness_contrast(
    image: Image,
    *,
    brightness: object = 0.0,
    contrast: object = 0.0,
    strength: object = 1.0,
) -> Image:
    """Classic brightness shift and contrast factor, both declared in [-1, 1]."""
    shift = _unit(brightness, "brightness", low=-1.0)
    factor = _unit(contrast, "contrast", low=-1.0)
    scale = (259.0 * (factor * 255.0 + 255.0)) / (255.0 * (259.0 - factor * 255.0))

    def convert(pixel: Rgb) -> Rgb:
        return tuple(  # type: ignore[return-value]
            clamp_rgb(scale * (channel + shift * 255.0 - 128.0) + 128.0)
            for channel in pixel
        )

    return _map(image, convert, strength)


def saturation(image: Image, *, factor: object = 1.0, strength: object = 1.0) -> Image:
    """Scale the distance from the grey level of each pixel."""
    amount = _unit(factor, "factor", high=4.0)

    def convert(pixel: Rgb) -> Rgb:
        grey = _luma(pixel)
        return tuple(  # type: ignore[return-value]
            clamp_rgb(grey + (channel - grey) * amount) for channel in pixel
        )

    return _map(image, convert, strength)


def tint(image: Image, *, factors: object = (1.0, 1.0, 1.0), strength: object = 1.0) -> Image:
    """Multiply each channel by a declared factor in [0, 4]."""
    if not isinstance(factors, (list, tuple)) or len(factors) != 3:
        raise ValueError("factors must list three channel multipliers")
    multipliers = tuple(
        _unit(value, f"factors[{index}]", high=4.0) for index, value in enumerate(factors)
    )

    def convert(pixel: Rgb) -> Rgb:
        return (
            clamp_rgb(pixel[0] * multipliers[0]),
            clamp_rgb(pixel[1] * multipliers[1]),
            clamp_rgb(pixel[2] * multipliers[2]),
        )

    return _map(image, convert, strength)


def threshold(image: Image, *, level: object = 0.5, strength: object = 1.0) -> Image:
    """Black and white at a declared luma level: a distinct filter, not a grayscale."""
    cut = _unit(level, "level")

    def convert(pixel: Rgb) -> Rgb:
        value = 255 if _luma(pixel) / 255.0 >= cut else 0
        return (value, value, value)

    return _map(image, convert, strength)


FILTERS = {
    "grayscale": grayscale,
    "sepia": sepia,
    "invert": invert,
    "brightness_contrast": brightness_contrast,
    "saturation": saturation,
    "tint": tint,
    "threshold": threshold,
}


def apply_filter(image: Image, name: str, **parameters: object) -> Image:
    """Apply one named filter with declared parameters, refusing unknown names."""
    if not isinstance(name, str) or name not in FILTERS:
        known = ", ".join(sorted(FILTERS))
        raise ValueError(f"unknown filter '{name}'; known filters: {known}")
    return FILTERS[name](image, **parameters)
