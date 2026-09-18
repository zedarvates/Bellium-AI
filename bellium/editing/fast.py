"""Faster filter paths that must agree with the plain implementation pixel for pixel.

Three backends exist: a vectorised path when NumPy is importable, a per-channel
lookup table path for the filters that are channel independent, and the plain
per-pixel implementation otherwise. NumPy stays optional: this module never
imports it at package import time and never requires it.
"""

from __future__ import annotations

from functools import lru_cache

from bellium.editing.filters import (
    FILTERS,
    LUMA_DIVISOR,
    LUMA_SCALED,
    SEPIA_DIVISOR,
    SEPIA_SCALED,
    _unit,
    apply_filter,
)
from bellium.knn._image import Image, shape

# Filters whose output channel depends only on the same input channel.
CHANNEL_WISE = ("invert", "brightness_contrast", "tint")


@lru_cache(maxsize=1)
def numpy_module():
    """The optional NumPy module, or None when it is not installed."""
    try:
        import numpy  # noqa: PLC0415
    except Exception:  # pragma: no cover - depends on the environment
        return None
    return numpy


def numpy_available() -> bool:
    return numpy_module() is not None


def backend_for(name: str, **parameters: object) -> str:
    """Declared backend for one filter call, so a caller can see what will run."""
    if name not in FILTERS:
        known = ", ".join(sorted(FILTERS))
        raise ValueError(f"unknown filter '{name}'; known filters: {known}")
    if numpy_available():
        return "numpy"
    if name in CHANNEL_WISE:
        return "lut"
    return "pure"


def channel_tables(name: str, **parameters: object) -> tuple[list[int], ...]:
    """256-entry tables for a channel-wise filter, built from the plain function."""
    if name not in CHANNEL_WISE:
        raise ValueError(f"{name} is not channel-wise; no lookup table exists")
    base = [[(level, level, level)] for level in range(256)]
    filtered = apply_filter(base, name, **parameters)
    tables = (
        [filtered[level][0][0] for level in range(256)],
        [filtered[level][0][1] for level in range(256)],
        [filtered[level][0][2] for level in range(256)],
    )
    return tables


def apply_with_tables(image: Image, tables: tuple[list[int], ...]) -> Image:
    """Apply channel tables, which is the same arithmetic without the per-pixel maths."""
    shape(image)
    red, green, blue = tables
    return [
        [(red[pixel[0]], green[pixel[1]], blue[pixel[2]]) for pixel in row]
        for row in image
    ]


def _numpy_image(image: Image):
    numpy = numpy_module()
    shape(image)
    return numpy.asarray(image, dtype=numpy.float64)


def _numpy_finish(array) -> Image:
    numpy = numpy_module()
    clipped = numpy.clip(numpy.round(array), 0.0, 255.0).astype(numpy.uint8)
    return [
        [(int(pixel[0]), int(pixel[1]), int(pixel[2])) for pixel in row]
        for row in clipped.tolist()
    ]


def _numpy_filter(array, name: str, **parameters: object):
    numpy = numpy_module()
    strength = _unit(parameters.get("strength", 1.0), "strength")
    if name == "grayscale":
        weights = parameters.get("weights", "luma")
        if weights not in ("luma", "average"):
            raise ValueError("weights must be luma or average")
        if weights == "luma":
            level = (array @ numpy.asarray(LUMA_SCALED, dtype=float)) / LUMA_DIVISOR
        else:
            level = array.mean(axis=2)
        stacked = numpy.stack([level, level, level], axis=2)
    elif name == "sepia":
        matrix = numpy.asarray(SEPIA_SCALED, dtype=float)
        stacked = (array @ matrix.T) / SEPIA_DIVISOR
    elif name == "invert":
        stacked = 255.0 - array
    elif name == "brightness_contrast":
        shift = _unit(parameters.get("brightness", 0.0), "brightness", low=-1.0)
        factor = _unit(parameters.get("contrast", 0.0), "contrast", low=-1.0)
        scale = (259.0 * (factor * 255.0 + 255.0)) / (255.0 * (259.0 - factor * 255.0))
        stacked = scale * (array + shift * 255.0 - 128.0) + 128.0
    elif name == "saturation":
        amount = _unit(parameters.get("factor", 1.0), "factor", high=4.0)
        grey = (
            (array @ numpy.asarray(LUMA_SCALED, dtype=float)) / LUMA_DIVISOR
        )[:, :, None]
        stacked = grey + (array - grey) * amount
    elif name == "tint":
        factors = parameters.get("factors", (1.0, 1.0, 1.0))
        if not isinstance(factors, (list, tuple)) or len(factors) != 3:
            raise ValueError("factors must list three channel multipliers")
        multipliers = numpy.asarray(
            [_unit(value, "factors", high=4.0) for value in factors], dtype=float
        )
        stacked = array * multipliers[None, None, :]
    elif name == "threshold":
        cut = _unit(parameters.get("level", 0.5), "level")
        level = (
            (array @ numpy.asarray(LUMA_SCALED, dtype=float)) / LUMA_DIVISOR / 255.0
        ) >= cut
        stacked = numpy.where(level[:, :, None], 255.0, 0.0)
        stacked = numpy.repeat(stacked, 3, axis=2) if stacked.shape[2] == 1 else stacked
    else:  # pragma: no cover - guarded by backend_for
        raise ValueError(f"no vectorised path for '{name}'")
    # The plain path rounds the filtered pixel before blending, so this path must
    # round in the same order to stay pixel for pixel identical.
    # The plain path clamps the filtered pixel before blending, so this path must
    # clamp in the same place or a saturated colour blends differently.
    filtered = numpy.clip(numpy.round(stacked), 0.0, 255.0)
    if strength < 1.0:
        filtered = numpy.round(array + (filtered - array) * strength)
    return filtered


def apply_filter_fast(image: Image, name: str, **parameters: object) -> Image:
    """Apply a filter through the fastest available path, matching the plain result."""
    if name not in FILTERS:
        known = ", ".join(sorted(FILTERS))
        raise ValueError(f"unknown filter '{name}'; known filters: {known}")
    if numpy_available():
        return _numpy_finish(_numpy_filter(_numpy_image(image), name, **parameters))
    if name in CHANNEL_WISE:
        return apply_with_tables(image, channel_tables(name, **parameters))
    return apply_filter(image, name, **parameters)


def downsample_fast(image: Image, *, max_side: int) -> dict:
    """Array-based area average, the same maths as the plain downsampler."""
    numpy = numpy_module()
    if numpy is None:
        raise ValueError("the fast downsampler needs NumPy")
    height, width = shape(image)
    if isinstance(max_side, bool) or not isinstance(max_side, int) or max_side < 4:
        raise ValueError("max_side must be an integer of at least four")
    if max(height, width) <= max_side:
        return {"image": [list(row) for row in image], "scale": 1.0,
                "size": (width, height), "resampled": False}
    scale = max_side / float(max(height, width))
    target_width = max(1, int(round(width * scale)))
    target_height = max(1, int(round(height * scale)))
    array = numpy.asarray(image, dtype=numpy.float64)
    row_edges = numpy.floor(
        numpy.arange(target_height + 1) * height / target_height
    ).astype(int)
    column_edges = numpy.floor(
        numpy.arange(target_width + 1) * width / target_width
    ).astype(int)
    row_edges[-1] = height
    column_edges[-1] = width
    rows = numpy.add.reduceat(array, numpy.maximum(row_edges[:-1], 0), axis=0)
    cells = numpy.add.reduceat(rows, numpy.maximum(column_edges[:-1], 0), axis=1)
    counts = numpy.outer(
        numpy.diff(row_edges), numpy.diff(column_edges)
    )[:, :, None]
    reduced = numpy.clip(numpy.round(cells / counts), 0.0, 255.0).astype(numpy.uint8)
    return {
        "image": [
            [(int(pixel[0]), int(pixel[1]), int(pixel[2])) for pixel in row]
            for row in reduced.tolist()
        ],
        "scale": round(scale, 6),
        "size": (target_width, target_height),
        "resampled": True,
    }


def to_array(image: Image):
    """One conversion into the array representation, to be reused across previews."""
    numpy = numpy_module()
    if numpy is None:
        raise ValueError("the array path needs NumPy")
    shape(image)
    return numpy.asarray(image, dtype=numpy.float64)


def from_array(array) -> Image:
    numpy = numpy_module()
    clipped = numpy.clip(numpy.round(array), 0.0, 255.0).astype(numpy.uint8)
    return [
        [(int(pixel[0]), int(pixel[1]), int(pixel[2])) for pixel in row]
        for row in clipped.tolist()
    ]


def reduce_array(array, *, max_side: int, round_values: bool = True):
    """Area average in array space; returns the reduced array and its geometry.

    round_values reproduces the plain downsampler, which returns integer pixels.
    A mask keeps its fractions, so it is reduced with round_values false.
    """
    numpy = numpy_module()
    if numpy is None:
        raise ValueError("the array path needs NumPy")
    height, width = array.shape[:2]
    if isinstance(max_side, bool) or not isinstance(max_side, int) or max_side < 4:
        raise ValueError("max_side must be an integer of at least four")
    if max(height, width) <= max_side:
        return array, 1.0, (width, height), False
    scale = max_side / float(max(height, width))
    target_width = max(1, int(round(width * scale)))
    target_height = max(1, int(round(height * scale)))
    row_edges = numpy.floor(
        numpy.arange(target_height + 1) * height / target_height
    ).astype(int)
    column_edges = numpy.floor(
        numpy.arange(target_width + 1) * width / target_width
    ).astype(int)
    row_edges[-1] = height
    column_edges[-1] = width
    rows = numpy.add.reduceat(array, numpy.maximum(row_edges[:-1], 0), axis=0)
    cells = numpy.add.reduceat(rows, numpy.maximum(column_edges[:-1], 0), axis=1)
    counts = numpy.outer(numpy.diff(row_edges), numpy.diff(column_edges))
    if array.ndim == 3:
        counts = counts[:, :, None]
    reduced = cells / counts
    if round_values:
        reduced = numpy.clip(numpy.round(reduced), 0.0, 255.0)
    return reduced, round(scale, 6), (target_width, target_height), True


def filter_array(array, name: str, **parameters: object):
    """Run a filter on the array representation without converting back."""
    if name not in FILTERS:
        known = ", ".join(sorted(FILTERS))
        raise ValueError(f"unknown filter '{name}'; known filters: {known}")
    if numpy_module() is None:
        raise ValueError("the array path needs NumPy")
    return _numpy_filter(array, name, **parameters)
