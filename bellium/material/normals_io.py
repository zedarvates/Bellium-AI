"""Normal maps as pixels: the encoding, the handedness and the decidable symptoms.

A normal map is a picture of a field, so reading one means decoding it with a known
encoding and a tangent-space handedness. The handedness is usually decidable and
sometimes not, and the difference is measured rather than assumed: negating the
green channel without mirroring the domain is not the gradient of any surface, so
the flipped reading carries a curl of twice the cross derivative of the x slope.
Where the surface bends, the two readings separate by 9 to 55 times and the map
decides itself; where the x slope is constant along the image vertical, or where
noise drowns the signal, both readings are valid surfaces and the module abstains.
The first version of this file asserted that the handedness was never inferable,
which the measurement contradicted.
"""

from __future__ import annotations

import math

from bellium.knn._image import Image, shape
from bellium.material.integration import CURL_LIMIT, gradients, integrability

Normals = list[list[tuple[float, float, float] | None]]
Flats = list[list[float]]
CONVENTIONS = ("+y", "-y")
NEAR_FLAT_PIXEL = (128, 128, 255)
NON_UNIT_LIMIT = 0.2
NEGATIVE_Z_LIMIT = 0.2
MEAN_Z_LIMIT = 0.2
# Measured margin between the two readings: a field whose x slope varies along the
# image vertical separated by 9.0, 15.1, 54.7 and 36.7, while a perturbed plane
# separated by 1.1 and an exact plane by nothing at all.
CONVENTION_MARGIN = 2.0


def _channel(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if not 0 <= value <= 255:
        raise ValueError(f"{name} must be between 0 and 255")
    return value


def _convention(name: object) -> str:
    if name not in CONVENTIONS:
        raise ValueError(f"convention must be one of: {', '.join(CONVENTIONS)}")
    return str(name)


def _normal(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _encode_channel(value: float) -> int:
    return min(255, max(0, int(round((value + 1.0) * 0.5 * 255.0))))


def encode_normal_map(normals: Normals, *, convention: str = "+y") -> Image:
    """Encode a normal field as an 8-bit RGB image.

    The mapping is the published one, n = 2 * c - 1, with no sRGB transfer: a
    normal map is data, not colour. Every normal is normalized first, so a field
    that is not unit length is stored as its direction and the loss is not hidden.
    """
    declaration = _convention(convention)
    if not isinstance(normals, list) or not normals or not normals[0]:
        raise ValueError("normals are empty")
    width = len(normals[0])
    rows = []
    for index, row in enumerate(normals):
        if len(row) != width:
            raise ValueError("normal rows must have equal width")
        pixels = []
        for column, normal in enumerate(row):
            if normal is None:
                pixels.append(NEAR_FLAT_PIXEL)
                continue
            if len(normal) != 3:
                raise ValueError("a normal needs three components")
            values = [_normal(normal[i], f"normal[{index}][{column}][{i}]") for i in range(3)]
            length = math.sqrt(sum(value * value for value in values))
            if length <= 1e-12:
                pixels.append(NEAR_FLAT_PIXEL)
                continue
            direction = [value / length for value in values]
            if declaration == "-y":
                direction[1] = -direction[1]
            pixels.append(tuple(_encode_channel(value) for value in direction))
        rows.append(pixels)
    return rows


def flip_normal_convention(image: Image) -> Image:
    """Swap the handedness: the green channel is the one that flips."""
    shape(image)
    return [
        [(pixel[0], 255 - _channel(pixel[1], "green"), pixel[2]) for pixel in row]
        for row in image
    ]


def decode_normal_map(
    image: Image,
    *,
    convention: str = "+y",
    z_from_xy: bool = True,
    mask: Flats | None = None,
) -> tuple[Normals, dict]:
    """Decode an 8-bit RGB image into unit normals, with what the decode noticed.

    With z_from_xy the blue channel is ignored and Z is reconstructed as
    sqrt(1 - x^2 - y^2), which is what a two-channel map needs and what the
    measured round trip in the gate record costs. Otherwise the stored blue
    channel is used and the vector is renormalized. A mask excludes pixels from
    the decode entirely: without it, the flat pixels that encode a missing normal
    are counted as surface and drag every statistic towards zero.
    """
    declaration = _convention(convention)
    rows, columns = shape(image)
    if mask is not None:
        if len(mask) != rows or any(len(row) != columns for row in mask):
            raise ValueError("the mask must share the image shape")
    normals: Normals = [[None] * columns for _ in range(rows)]
    pixels = 0
    unreconstructable = 0
    non_unit = 0.0
    flat = 0
    z_total = 0.0
    for r in range(rows):
        for c in range(columns):
            if mask is not None and mask[r][c] <= 0.0:
                continue
            pixel = image[r][c]
            if not isinstance(pixel, tuple) or len(pixel) != 3:
                raise ValueError(f"pixel [{r}][{c}] must be three channels")
            red = _channel(pixel[0], "red")
            green = _channel(pixel[1], "green")
            blue = _channel(pixel[2], "blue")
            x = red / 255.0 * 2.0 - 1.0
            y = green / 255.0 * 2.0 - 1.0
            if declaration == "-y":
                y = -y
            if z_from_xy:
                squared = x * x + y * y
                if squared > 1.0:
                    unreconstructable += 1
                z = math.sqrt(max(0.0, 1.0 - squared))
            else:
                z = blue / 255.0 * 2.0 - 1.0
            length = math.sqrt(x * x + y * y + z * z)
            if length <= 1e-9:
                flat += 1
                continue
            non_unit += abs(length - 1.0)
            z_total += z / length
            normals[r][c] = (x / length, y / length, z / length)
            pixels += 1
    return normals, {
        "convention": declaration,
        "z_source": "reconstructed" if z_from_xy else "stored",
        "pixels": pixels,
        "unreconstructable": unreconstructable,
        "mean_non_unit": round(non_unit / pixels, 6) if pixels else None,
        "mean_z": round(z_total / pixels, 6) if pixels else None,
        "flat_pixels": flat,
    }


def inspect_normal_map(
    image: Image, *, convention: str = "+y", mask: Flats | None = None
) -> dict:
    """Report what a map states about itself, and what it cannot state.

    The verdicts are the decidable ones, measured in the gate record: a field
    whose vectors are not unit length is not a normal map, an object-space map
    carries negative Z across half of it, an inverted-Z map is one whose mean Z is
    negative. The handedness is not on that list, and the answer says so.
    """
    rows, columns = shape(image)
    normals, report = decode_normal_map(
        image, convention=convention, z_from_xy=False, mask=mask
    )
    detection = detect_convention(image, mask=mask)
    if not report["pixels"]:
        return {
            "status": "abstain",
            "reason": "no_normal_pixels",
            "verdict": None,
            "convention": report["convention"],
            "y_convention_decidable": False,
            "y_convention_detection": detection,
            "decode": report,
            "saturation_ratio": 0.0,
            "negative_z_ratio": 0.0,
        }
    negative = 0
    saturation = 0
    counted_pixels = 0
    for r in range(rows):
        for c in range(columns):
            if mask is not None and mask[r][c] <= 0.0:
                continue
            normal = normals[r][c]
            if normal is None:
                continue
            counted_pixels += 1
            if normal[2] < 0.0:
                negative += 1
            pixel = image[r][c]
            if any(value in (0, 255) for value in pixel):
                saturation += 1
    negative_ratio = negative / report["pixels"]
    saturation_ratio = saturation / counted_pixels if counted_pixels else 0.0
    mean_z = report["mean_z"] or 0.0
    if (report["mean_non_unit"] or 0.0) > NON_UNIT_LIMIT:
        verdict = "not-a-normal-map"
    elif negative_ratio > 0.8 and mean_z < -MEAN_Z_LIMIT:
        # The whole map faces away: an inverted-Z asset, not an object-space one.
        verdict = "inverted-z"
    elif negative_ratio > NEGATIVE_Z_LIMIT:
        verdict = "object-space"
    elif mean_z > MEAN_Z_LIMIT:
        verdict = "tangent-space"
    else:
        verdict = "ambiguous"
    return {
        "status": "ready",
        "reason": None,
        "verdict": verdict,
        "convention": report["convention"],
        "mean_z": report["mean_z"],
        "negative_z_ratio": round(negative_ratio, 6),
        "mean_non_unit": report["mean_non_unit"],
        "saturation_ratio": round(saturation_ratio, 6),
        "pixels": report["pixels"],
        "decode": report,
        "y_convention_decidable": detection["status"] == "decided",
        "y_convention_detection": detection,
    }


def detect_convention(image: Image, *, mask: Flats | None = None) -> dict:
    """Decide the handedness from integrability, or say why it cannot be decided.

    Both readings of the picture are decoded and integrated in the sense that
    matters here: the interior curl of each slope field is measured. The reading
    with the lower curl is the one that is a gradient of a surface; the other is
    not a gradient at all unless the x slope happens to be constant along the image
    vertical.

    The absolute curl is what is compared, so an inverted blue channel, which
    flips every slope's sign, does not change the answer. The scale of the two
    readings is irrelevant for the same reason, which is why no pixel scale is
    needed.
    """
    readings: dict[str, dict] = {}
    for declaration in CONVENTIONS:
        normals, _ = decode_normal_map(image, convention=declaration, z_from_xy=True)
        slope_x, slope_y, valid = gradients(normals, mask)
        readings[declaration] = integrability(slope_x, slope_y, valid)
    plus = readings["+y"]["curl_ratio"]
    minus = readings["-y"]["curl_ratio"]
    shared = {
        "readings": {
            name: {
                "curl_ratio": value["curl_ratio"],
                "verdict": value["verdict"],
                "cells": value["cells"],
            }
            for name, value in readings.items()
        },
        "margin_limit": CONVENTION_MARGIN,
        "integrable_limit": CURL_LIMIT,
    }
    if plus is None or minus is None:
        return {
            **shared,
            "status": "abstain",
            "reason": "no_interior_pixels",
            "convention": None,
            "margin": None,
        }
    best, worst = min(plus, minus), max(plus, minus)
    if worst <= CURL_LIMIT:
        return {
            **shared,
            "status": "abstain",
            "reason": "both_readings_integrable",
            "convention": None,
            "margin": None,
        }
    margin = worst / best if best > 1e-9 else None
    if margin is None or margin >= CONVENTION_MARGIN:
        return {
            **shared,
            "status": "decided",
            "reason": None,
            "convention": "+y" if plus <= minus else "-y",
            "margin": None if margin is None else round(margin, 4),
        }
    return {
        **shared,
        "status": "abstain",
        "reason": "margin_below_declared",
        "convention": None,
        "margin": round(margin, 4),
    }


def quantize_height(
    height: Flats,
    valid: Flats,
    *,
    bit_depth: int = 8,
    span: tuple[float, float] | None = None,
) -> dict:
    """Quantize a height map for a displacement channel, with the loss reported.

    Heights are relative, so the span is either declared by the caller or taken
    from the valid pixels, and the returned step and worst error state what the
    quantization cost. No file is written.
    """
    if isinstance(bit_depth, bool) or not isinstance(bit_depth, int) or not 1 <= bit_depth <= 16:
        raise ValueError("bit_depth must be an integer between 1 and 16")
    rows = len(height)
    columns = len(height[0])
    for row in height:
        if len(row) != columns:
            raise ValueError("height rows must have equal width")
    if len(valid) != rows or any(len(row) != columns for row in valid):
        raise ValueError("the mask must share the height shape")
    samples = [
        float(height[r][c]) for r in range(rows) for c in range(columns) if valid[r][c]
    ]
    if not samples:
        return {
            "values": [[0] * columns for _ in range(rows)],
            "valid": valid,
            "bit_depth": bit_depth,
            "span": None,
            "step": None,
            "worst_error": None,
            "pixels": 0,
        }
    if span is None:
        low, high = min(samples), max(samples)
    else:
        if not isinstance(span, tuple) or len(span) != 2:
            raise ValueError("span must be a pair of numbers")
        low, high = (_normal(span[0], "span[0]"), _normal(span[1], "span[1]"))
        if high < low:
            raise ValueError("span must not be inverted")
    levels = (1 << bit_depth) - 1
    step = (high - low) / levels if levels else 0.0
    values = [[0] * columns for _ in range(rows)]
    worst = 0.0
    for r in range(rows):
        for c in range(columns):
            if not valid[r][c]:
                continue
            value = float(height[r][c])
            if step > 0.0:
                level = min(levels, max(0, int(round((value - low) / step))))
            else:
                level = 0
            values[r][c] = level
            worst = max(worst, abs(level * step + low - value))
    return {
        "values": values,
        "valid": valid,
        "bit_depth": bit_depth,
        "span": [round(low, 6), round(high, 6)],
        "step": round(step, 8),
        "worst_error": round(worst, 8),
        "pixels": len(samples),
    }
