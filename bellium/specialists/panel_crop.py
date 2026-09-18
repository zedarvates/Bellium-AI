"""Panel-safe crop: never clip required content. k-NN only proposes margin."""

from __future__ import annotations

from typing import Any

from bellium.contracts.schema import AuthorityMode, SpecialistResult
from bellium.knn._image import Image, shape
from bellium.knn.color_cutout import cutout
from bellium.knn.panel_crop import FAMILIES, layout_features, propose_margin

SPECIALIST_ID = "bellium/hybrid/panel-safe-crop:v0"
FALLBACK_MARGIN = {
    "storycore-panel": 0.10,
    "sprite": 0.06,
    "product": 0.12,
}


def _box(raw: object, width: int, height: int) -> tuple[int, int, int, int]:
    if not isinstance(raw, dict):
        raise ValueError("content box must be an object")
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
        raise ValueError("content box sits outside the image")
    return x, y, w, h


def _union(boxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    x0 = min(box[0] for box in boxes)
    y0 = min(box[1] for box in boxes)
    x1 = max(box[0] + box[2] for box in boxes)
    y1 = max(box[1] + box[3] for box in boxes)
    return x0, y0, x1 - x0, y1 - y0


def _from_alpha(alpha: list[list[float]]) -> tuple[int, int, int, int] | None:
    rows = [r for r, row in enumerate(alpha) if any(row)]
    cols = [c for c in range(len(alpha[0])) if any(alpha[r][c] for r in range(len(alpha)))]
    if not rows or not cols:
        return None
    x0, x1 = min(cols), max(cols)
    y0, y1 = min(rows), max(rows)
    return x0, y0, x1 - x0 + 1, y1 - y0 + 1


def _touch_flags(box: tuple[int, int, int, int], width: int, height: int) -> dict[str, float]:
    x, y, w, h = box
    return {
        "touch_left": 1.0 if x <= 0 else 0.0,
        "touch_right": 1.0 if x + w >= width else 0.0,
        "touch_top": 1.0 if y <= 0 else 0.0,
        "touch_bottom": 1.0 if y + h >= height else 0.0,
    }


def _features(box: tuple[int, int, int, int], width: int, height: int) -> dict[str, float]:
    x, y, w, h = box
    fill = (w * h) / (width * height)
    aspect = min(w, h) / max(w, h)
    feats = {"fill": fill, "aspect": aspect}
    feats.update(_touch_flags(box, width, height))
    return layout_features(feats)


def _apply_margin(box: tuple[int, int, int, int], width: int, height: int, margin: float) -> tuple[int, int, int, int]:
    pad_x = int(round(width * margin))
    pad_y = int(round(height * margin))
    x0 = max(0, box[0] - pad_x)
    y0 = max(0, box[1] - pad_y)
    x1 = min(width, box[0] + box[2] + pad_x)
    y1 = min(height, box[1] + box[3] + pad_y)
    return x0, y0, x1 - x0, y1 - y0


def _slice(image: Image, box: tuple[int, int, int, int]) -> Image:
    x, y, w, h = box
    return [row[x:x + w] for row in image[y:y + h]]


def crop_panel(query: dict[str, Any]) -> SpecialistResult:
    image = query.get("image")
    if image is None:
        raise ValueError("query needs image")
    height, width = shape(image)
    family = str(query.get("family") or "").strip()
    if family not in FAMILIES:
        return SpecialistResult(
            SPECIALIST_ID,
            {"status": "abstain", "reason": "unknown_family", "family": family},
            0.0, True, AuthorityMode.CONSULTATIVE,
            notes=("Unknown layout family; no crop is invented.",),
        )
    raw_boxes = query.get("content_boxes")
    if raw_boxes is None:
        inferred = cutout(image)
        if inferred.abstained:
            return SpecialistResult(
                SPECIALIST_ID,
                {"status": "abstain", "reason": "content_not_found", "cutout": inferred.output},
                0.0, True, AuthorityMode.CONSULTATIVE,
                notes=("Without boxes, crop needs a simple foreground.",),
            )
        content = _from_alpha(inferred.output["alpha"])
        if content is None:
            return SpecialistResult(
                SPECIALIST_ID,
                {"status": "abstain", "reason": "empty_foreground"},
                0.0, True, AuthorityMode.CONSULTATIVE,
            )
        required = [content]
    else:
        if not isinstance(raw_boxes, list) or not raw_boxes:
            raise ValueError("content_boxes must be a non-empty list")
        required = [_box(item, width, height) for item in raw_boxes]
        content = _union(required)
    touches = _touch_flags(content, width, height)
    already_clipped = any(touches.values())
    if already_clipped and raw_boxes is not None:
        return SpecialistResult(
            SPECIALIST_ID,
            {
                "status": "unsafe",
                "reason": "required_content_already_clipped",
                "family": family,
                "content_box": {"x": content[0], "y": content[1], "w": content[2], "h": content[3]},
            },
            1.0, False, AuthorityMode.CONSULTATIVE,
            notes=("A required box already touches the frame. Cropping cannot repair it.",),
        )
    knn = propose_margin({"family": family, "features": _features(content, width, height)})
    if knn.output.get("verdict") == "unsafe":
        return SpecialistResult(
            SPECIALIST_ID,
            {"status": "unsafe", "reason": "neighbors_look_clipped", "family": family,
             "knn": knn.output},
            knn.confidence, False, AuthorityMode.CONSULTATIVE,
            notes=("k-NN neighbors look like clipped panels.",),
        )
    margin = knn.output.get("margin")
    if margin is None:
        margin = FALLBACK_MARGIN[family]
    crop = _apply_margin(content, width, height, float(margin))
    for x, y, w, h in required:
        if x < crop[0] or y < crop[1] or x + w > crop[0] + crop[2] or y + h > crop[1] + crop[3]:
            return SpecialistResult(
                SPECIALIST_ID,
                {"status": "unsafe", "reason": "crop_would_clip_content", "family": family},
                1.0, False, AuthorityMode.CONSULTATIVE,
            )
    sliced = _slice(image, crop)
    return SpecialistResult(
        SPECIALIST_ID,
        {
            "status": "safe",
            "family": family,
            "margin": round(float(margin), 4),
            "crop": {"x": crop[0], "y": crop[1], "w": crop[2], "h": crop[3]},
            "image": sliced,
            "knn": knn.output,
            "certified": False,
        },
        0.8 if not knn.abstained else 0.55,
        False, AuthorityMode.CONSULTATIVE,
        notes=("Consultative crop only. Not a comic parser.",),
    )

