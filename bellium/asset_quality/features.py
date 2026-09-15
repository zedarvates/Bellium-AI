"""Versioned alpha geometry, not aesthetic or semantic quality scores."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

from PIL import Image

FEATURE_SCHEMA = "bellium.alpha-features/v1"
FEATURE_NAMES = (
    "coverage", "partial_alpha", "border_coverage", "bbox_area",
    "horizontal_offset", "vertical_offset",
)
MAX_FILE_BYTES = 32 * 1024 * 1024
MAX_PIXELS = 16 * 1024 * 1024


def validate_vector(values: object) -> list[float]:
    if not isinstance(values, (list, tuple)) or len(values) != len(FEATURE_NAMES):
        raise ValueError("Expected six alpha features")
    if any(isinstance(v, bool) or not isinstance(v, (int, float))
           or not math.isfinite(v) or not 0 <= v <= 1 for v in values):
        raise ValueError("Features must be finite numbers in [0, 1]")
    return [float(v) for v in values]


def extract_features(image: Image.Image) -> list[float]:
    w, h = image.size
    if min(w, h) < 2 or w * h > MAX_PIXELS:
        raise ValueError("Image must be at least 2x2 and at most 16 megapixels")
    if "A" not in image.getbands() and "transparency" not in image.info:
        raise ValueError("An explicit alpha channel is required; run cutout first")
    alpha = image.convert("RGBA").getchannel("A")
    histogram = alpha.histogram()
    count = w * h
    coverage = sum(histogram[129:]) / count
    partial = sum(histogram[16:240]) / count
    solid = alpha.point(lambda value: 255 if value > 128 else 0)
    bbox = solid.getbbox()
    border_count = sum(solid.crop((0, 0, w, 1)).histogram()[1:])
    border_count += sum(solid.crop((0, h - 1, w, h)).histogram()[1:])
    if h > 2:
        border_count += sum(solid.crop((0, 1, 1, h - 1)).histogram()[1:])
        border_count += sum(solid.crop((w - 1, 1, w, h - 1)).histogram()[1:])
    if bbox is None:
        area = horizontal = vertical = 0.0
    else:
        left, top, right, bottom = bbox
        area = (right - left) * (bottom - top) / count
        horizontal = abs((left + right) / w - 1)
        vertical = abs((top + bottom) / h - 1)
    return validate_vector([
        coverage, partial, border_count / (2 * w + 2 * h - 4),
        area, horizontal, vertical,
    ])


def read_asset(path: str | Path) -> tuple[Image.Image, str, int]:
    """Read once: hash the exact bytes decoded, rejecting oversized/multiframe input."""
    import io
    path = Path(path)
    with path.open("rb") as stream:
        payload = stream.read(MAX_FILE_BYTES + 1)
    if not payload or len(payload) > MAX_FILE_BYTES:
        raise ValueError("Image must be nonempty and at most 32 MiB")
    with Image.open(io.BytesIO(payload)) as source:
        if source.width * source.height > MAX_PIXELS or min(source.size) < 2:
            raise ValueError("Image must be at least 2x2 and at most 16 megapixels")
        if getattr(source, "n_frames", 1) != 1:
            raise ValueError("Animated/multiframe assets need a separate validator")
        source.load()
        image = source.copy()
    return image, hashlib.sha256(payload).hexdigest(), len(payload)


def rule_advice(values: list[float]) -> dict:
    coverage, partial, border, area, horizontal, vertical = validate_vector(values)
    reasons = []
    if coverage < 0.02:
        reasons.append("empty_or_tiny_foreground")
    if coverage > 0.95:
        reasons.append("almost_opaque_canvas")
    if partial > 0.12:
        reasons.append("broad_partial_alpha")
    if border > 0.0:
        reasons.append("foreground_touches_canvas")
    if area < 0.04:
        reasons.append("small_bounding_box")
    if max(horizontal, vertical) > 0.45:
        reasons.append("off_center_foreground")
    return {"label": "review" if reasons else "candidate", "reasons": reasons}
