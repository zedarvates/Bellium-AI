"""Find plausible speech-bubble regions. Never reads or stores dialogue."""

from __future__ import annotations

from typing import Any

from bellium.contracts.schema import AuthorityMode, SpecialistResult
from bellium.knn._image import Image, shape
from bellium.knn.speech_bubble import classify_region

SPECIALIST_ID = "bellium/hybrid/speech-bubble-region:v0"
MIN_FILL = 0.02
MAX_FILL = 0.45
MIN_COMPACT = 0.45
LUMA_GATE = 0.82


def _luma(pixel: tuple[int, int, int]) -> float:
    return (0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.0722 * pixel[2]) / 255.0


def _box(raw: object, width: int, height: int) -> tuple[int, int, int, int]:
    if not isinstance(raw, dict):
        raise ValueError("region box must be an object")
    for key in ("x", "y", "w", "h"):
        if raw.get(key) is None:
            raise ValueError(f"box field {key} is unknown; do not coerce to 0")
        value = raw[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"box field {key} must be a non-negative integer")
    x, y, w, h = int(raw["x"]), int(raw["y"]), int(raw["w"]), int(raw["h"])
    if w == 0 or h == 0:
        raise ValueError("box needs positive width and height")
    if x + w > width or y + h > height:
        raise ValueError("region box sits outside the image")
    return x, y, w, h


def _components(mask: list[list[int]]) -> list[list[tuple[int, int]]]:
    height, width = len(mask), len(mask[0])
    seen = [[False] * width for _ in range(height)]
    blobs: list[list[tuple[int, int]]] = []
    for r in range(height):
        for c in range(width):
            if not mask[r][c] or seen[r][c]:
                continue
            stack = [(r, c)]
            seen[r][c] = True
            pixels = [(r, c)]
            while stack:
                cr, cc = stack.pop()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < height and 0 <= nc < width and mask[nr][nc] and not seen[nr][nc]:
                        seen[nr][nc] = True
                        stack.append((nr, nc))
                        pixels.append((nr, nc))
            blobs.append(pixels)
    return blobs


def region_features(image: Image, box: tuple[int, int, int, int]) -> dict[str, float]:
    height, width = shape(image)
    x, y, w, h = box
    area = w * h
    lumas = []
    bright = 0
    for r in range(y, y + h):
        for c in range(x, x + w):
            value = _luma(image[r][c])
            lumas.append(value)
            if value >= LUMA_GATE:
                bright += 1
    compactness = bright / area
    return {
        "fill": area / (width * height),
        "aspect": min(w, h) / max(w, h),
        "compactness": compactness,
        "brightness": sum(lumas) / len(lumas),
        "border_touch": 1.0 if x <= 0 or y <= 0 or x + w >= width or y + h >= height else 0.0,
        "topness": (y + h / 2) / height,
    }


def _from_blob(pixels: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in pixels]
    cols = [c for _, c in pixels]
    x0, x1 = min(cols), max(cols)
    y0, y1 = min(rows), max(rows)
    return x0, y0, x1 - x0 + 1, y1 - y0 + 1


def _detect(image: Image) -> list[tuple[int, int, int, int]]:
    height, width = shape(image)
    mask = [[1 if _luma(image[r][c]) >= LUMA_GATE else 0 for c in range(width)] for r in range(height)]
    found = []
    total = height * width
    for blob in _components(mask):
        box = _from_blob(blob)
        fill = (box[2] * box[3]) / total
        compact = len(blob) / (box[2] * box[3])
        if fill < MIN_FILL or fill > MAX_FILL:
            continue
        if compact < MIN_COMPACT:
            continue
        found.append(box)
    return found


def locate_bubbles(query: dict[str, Any]) -> SpecialistResult:
    if query.get("text") is not None or query.get("dialogue") is not None:
        raise ValueError("pass regions or an image, not dialogue text")
    image = query.get("image")
    if image is None:
        raise ValueError("query needs image")
    height, width = shape(image)
    raw_boxes = query.get("regions")
    if raw_boxes is None:
        boxes = _detect(image)
        origin = "detected"
    else:
        if not isinstance(raw_boxes, list) or not raw_boxes:
            raise ValueError("regions must be a non-empty list")
        boxes = [_box(item, width, height) for item in raw_boxes]
        origin = "provided"
    if not boxes:
        return SpecialistResult(
            SPECIALIST_ID,
            {"status": "abstain", "reason": "no_bright_compact_region", "regions": [], "text": None},
            0.0, True, AuthorityMode.CONSULTATIVE,
            notes=("No plausible bubble region. Dialogue is not read.",),
        )
    accepted = []
    rejected = []
    for box in boxes:
        features = region_features(image, box)
        verdict = classify_region({"features": features})
        record = {
            "x": box[0], "y": box[1], "w": box[2], "h": box[3],
            "label": verdict.output.get("label"),
            "knn": verdict.output,
        }
        if verdict.abstained or verdict.output.get("label") != "bubble":
            rejected.append(record)
        else:
            accepted.append(record)
    if not accepted:
        return SpecialistResult(
            SPECIALIST_ID,
            {"status": "abstain", "reason": "no_bubble_label", "origin": origin,
             "regions": [], "rejected": rejected, "text": None},
            0.0, True, AuthorityMode.CONSULTATIVE,
            notes=("Bright regions were found but not labelled as bubbles.",),
        )
    return SpecialistResult(
        SPECIALIST_ID,
        {
            "status": "suggest",
            "origin": origin,
            "regions": accepted,
            "rejected": rejected,
            "text": None,
            "certified": False,
        },
        0.7, False, AuthorityMode.CONSULTATIVE,
        notes=("Consultative bubble regions only. Not OCR and not a comic parser.",),
    )

