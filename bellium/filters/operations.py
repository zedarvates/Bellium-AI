"""Deterministic image filters with adjustable strength and optional masks.

Operations run on 8-bit sRGB pixels through Pillow, with no learned weights,
no randomness and no network access. Alpha is preserved when the input has an
alpha channel. Masks use mode "L" or "1", must match the image size, and blend
the filtered result over the source (255 = filtered, 0 = source).
"""
from __future__ import annotations

import math

from PIL import Image, ImageChops, ImageEnhance, ImageOps

FILTER_NAMES: tuple[str, ...] = (
    "grayscale",
    "sepia",
    "binary",
    "invert",
    "brightness",
    "contrast",
    "saturation",
    "tint",
)
SEPIA_DARK: tuple[int, int, int] = (32, 16, 8)
SEPIA_LIGHT: tuple[int, int, int] = (244, 223, 181)
MAX_FACTOR: float = 16.0


class FilterError(ValueError):
    """Raised when a filter request is invalid."""


def _validate_strength(strength: float) -> float:
    if isinstance(strength, bool) or not isinstance(strength, (int, float)):
        raise FilterError("strength must be a finite number between 0 and 1.")
    value = float(strength)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise FilterError("strength must be a finite number between 0 and 1.")
    return value


def _validate_threshold(threshold: int) -> int:
    if isinstance(threshold, bool) or not isinstance(threshold, int) or not 0 <= threshold <= 255:
        raise FilterError("threshold must be an integer between 0 and 255.")
    return threshold


def _validate_factor(factor: float) -> float:
    if isinstance(factor, bool) or not isinstance(factor, (int, float)):
        raise FilterError(f"factor must be a finite number between 0 and {MAX_FACTOR:g}.")
    value = float(factor)
    if not math.isfinite(value) or not 0.0 <= value <= MAX_FACTOR:
        raise FilterError(f"factor must be a finite number between 0 and {MAX_FACTOR:g}.")
    return value


def _validate_color(color: tuple[int, int, int], name: str) -> tuple[int, int, int]:
    try:
        channels = tuple(color)
    except TypeError as exc:
        raise FilterError(f"{name} must be an (R, G, B) tuple.") from exc
    if len(channels) != 3:
        raise FilterError(f"{name} must be an (R, G, B) tuple.")
    for channel in channels:
        if isinstance(channel, bool) or not isinstance(channel, int) or not 0 <= channel <= 255:
            raise FilterError(f"{name} channels must be integers between 0 and 255.")
    return (channels[0], channels[1], channels[2])


def _validate_mask(mask: Image.Image, size: tuple[int, int]) -> Image.Image:
    if not isinstance(mask, Image.Image):
        raise FilterError("mask must be a PIL.Image.Image instance.")
    if mask.size != size:
        raise FilterError("Mask and image must have identical dimensions.")
    if mask.mode not in ("1", "L"):
        raise FilterError("Mask must have mode '1' or 'L'.")
    return mask


def _has_alpha(image: Image.Image) -> bool:
    return image.mode in ("RGBA", "LA", "PA") or (
        image.mode == "P" and "transparency" in image.info
    )


def _restore_alpha(result: Image.Image, source: Image.Image) -> Image.Image:
    if not _has_alpha(source):
        return result
    output = result.convert("RGBA")
    output.putalpha(source.convert("RGBA").getchannel("A"))
    return output


def _apply(image: Image.Image, transform, *, strength, mask) -> Image.Image:
    if not isinstance(image, Image.Image):
        raise FilterError("image must be a PIL.Image.Image instance.")
    strength = _validate_strength(strength)
    if mask is not None:
        mask = _validate_mask(mask, image.size)
    base = image.convert("RGB")
    if strength == 0.0:
        result = base
    else:
        filtered = transform(base)
        result = filtered if strength == 1.0 else Image.blend(base, filtered, strength)
    if mask is not None:
        result = Image.composite(result, base, mask)
    return _restore_alpha(result, image)


def to_grayscale(
    image: Image.Image,
    *,
    strength: float = 1.0,
    mask: Image.Image | None = None,
) -> Image.Image:
    """Convert to luma grayscale (Pillow "L" mode: ITU-R 601-2 weights)."""
    return _apply(
        image,
        lambda rgb: ImageOps.grayscale(rgb).convert("RGB"),
        strength=strength,
        mask=mask,
    )


def to_sepia(
    image: Image.Image,
    *,
    strength: float = 1.0,
    mask: Image.Image | None = None,
    dark: tuple[int, int, int] = SEPIA_DARK,
    light: tuple[int, int, int] = SEPIA_LIGHT,
) -> Image.Image:
    """Map luma through a warm duotone, dark to light, with adjustable strength."""
    dark = _validate_color(dark, "dark")
    light = _validate_color(light, "light")
    black = "#%02x%02x%02x" % dark
    white = "#%02x%02x%02x" % light

    def transform(rgb: Image.Image) -> Image.Image:
        return ImageOps.colorize(ImageOps.grayscale(rgb), black=black, white=white)

    return _apply(image, transform, strength=strength, mask=mask)


def to_binary(
    image: Image.Image,
    *,
    threshold: int = 128,
    strength: float = 1.0,
    mask: Image.Image | None = None,
) -> Image.Image:
    """Threshold luma to black and white; values at or above threshold become white."""
    threshold = _validate_threshold(threshold)

    def transform(rgb: Image.Image) -> Image.Image:
        gray = ImageOps.grayscale(rgb)
        return gray.point(lambda value: 255 if value >= threshold else 0, mode="L").convert("RGB")

    return _apply(image, transform, strength=strength, mask=mask)


def invert(
    image: Image.Image,
    *,
    strength: float = 1.0,
    mask: Image.Image | None = None,
) -> Image.Image:
    """Invert every RGB channel (255 - value); alpha is preserved."""
    return _apply(image, ImageChops.invert, strength=strength, mask=mask)


def brightness(
    image: Image.Image,
    *,
    factor: float = 1.0,
    strength: float = 1.0,
    mask: Image.Image | None = None,
) -> Image.Image:
    """Scale luminance with Pillow ImageEnhance: 0 is black, 1 is identity."""
    factor = _validate_factor(factor)
    return _apply(
        image,
        lambda rgb: ImageEnhance.Brightness(rgb).enhance(factor),
        strength=strength,
        mask=mask,
    )


def contrast(
    image: Image.Image,
    *,
    factor: float = 1.0,
    strength: float = 1.0,
    mask: Image.Image | None = None,
) -> Image.Image:
    """Adjust contrast with Pillow ImageEnhance: 0 is flat gray, 1 is identity."""
    factor = _validate_factor(factor)
    return _apply(
        image,
        lambda rgb: ImageEnhance.Contrast(rgb).enhance(factor),
        strength=strength,
        mask=mask,
    )


def saturation(
    image: Image.Image,
    *,
    factor: float = 1.0,
    strength: float = 1.0,
    mask: Image.Image | None = None,
) -> Image.Image:
    """Adjust color saturation: 0 is grayscale, 1 is identity, above 1 is stronger."""
    factor = _validate_factor(factor)
    return _apply(
        image,
        lambda rgb: ImageEnhance.Color(rgb).enhance(factor),
        strength=strength,
        mask=mask,
    )


def tint(
    image: Image.Image,
    *,
    color: tuple[int, int, int] | None = None,
    strength: float = 1.0,
    mask: Image.Image | None = None,
) -> Image.Image:
    """Multiply RGB channels by a tint color, like a color gel; white is identity."""
    if color is None:
        raise FilterError("tint requires an (R, G, B) color.")
    color = _validate_color(color, "color")

    def transform(rgb: Image.Image) -> Image.Image:
        return ImageChops.multiply(rgb, Image.new("RGB", rgb.size, color))

    return _apply(image, transform, strength=strength, mask=mask)


def apply_filter(name: str, image: Image.Image, **options) -> Image.Image:
    """Dispatch any advertised filter name."""
    if name == "grayscale":
        return to_grayscale(image, **options)
    if name == "sepia":
        return to_sepia(image, **options)
    if name == "binary":
        return to_binary(image, **options)
    if name == "invert":
        return invert(image, **options)
    if name == "brightness":
        return brightness(image, **options)
    if name == "contrast":
        return contrast(image, **options)
    if name == "saturation":
        return saturation(image, **options)
    if name == "tint":
        return tint(image, **options)
    raise FilterError(f"Unknown filter {name!r}; available: {', '.join(FILTER_NAMES)}.")
