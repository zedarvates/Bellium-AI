"""Sprite frame preparation: crop to content, pad, and place the anchor.

The frame is built from a declared foreground mask. Content that already
touches the frame refuses preparation instead of being trimmed, and the source
image is never modified.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from bellium.contracts.schema import AuthorityMode, SpecialistResult
from bellium.knn._image import Image, Mask, shape, validate_mask
from bellium.knn.sprite_anchor import FAMILIES, propose_anchor

SPECIALIST_ID = "bellium/hybrid/sprite-frame-prep:v0"
DEFAULT_PAD_RATIO = 0.08
MAX_PAD_RATIO = 0.5
DEFAULT_ANCHOR = {
    "character": "feet",
    "prop": "feet",
    "effect": "center",
    "ui": "center",
}


def _bbox(mask: Mask) -> tuple[int, int, int, int] | None:
    rows = [r for r, row in enumerate(mask) if any(row)]
    cols = [c for c in range(len(mask[0])) if any(row[c] for row in mask)]
    if not rows or not cols:
        return None
    return min(cols), min(rows), max(cols), max(rows)


def _ratio(raw: object) -> float:
    if raw is None:
        return DEFAULT_PAD_RATIO
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError("pad_ratio must be numeric")
    value = float(raw)
    if not 0.0 <= value <= MAX_PAD_RATIO:
        raise ValueError(f"pad_ratio must be between 0 and {MAX_PAD_RATIO}")
    return value


def _color(raw: object) -> tuple[int, int, int] | None:
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise ValueError("pad_color must list three channels")
    channels = []
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 255:
            raise ValueError("pad_color channels must be integers in [0, 255]")
        channels.append(value)
    return channels[0], channels[1], channels[2]


def prepare_sprite_frame(query: dict[str, Any]) -> SpecialistResult:
    image = query.get("image")
    mask = query.get("mask")
    if image is None or mask is None:
        raise ValueError("query needs image and mask")
    validate_mask(image, mask)
    height, width = shape(image)
    family = str(query.get("family") or "").strip()
    if family not in FAMILIES:
        return SpecialistResult(
            SPECIALIST_ID,
            {"status": "abstain", "reason": "unknown_family", "family": family,
             "image": None},
            0.0, True, AuthorityMode.CONSULTATIVE,
            notes=("Unknown sprite family; no frame is invented.",),
        )
    box = _bbox(mask)
    if box is None:
        return SpecialistResult(
            SPECIALIST_ID,
            {"status": "abstain", "reason": "empty_foreground", "family": family,
             "image": None},
            0.0, True, AuthorityMode.CONSULTATIVE,
        )
    x0, y0, x1, y1 = box
    if x0 == 0 or y0 == 0 or x1 == width - 1 or y1 == height - 1:
        return SpecialistResult(
            SPECIALIST_ID,
            {
                "status": "unsafe",
                "reason": "content_touches_frame",
                "family": family,
                "image": None,
                "content_box": {"x": x0, "y": y0, "w": x1 - x0 + 1, "h": y1 - y0 + 1},
            },
            1.0, False, AuthorityMode.CONSULTATIVE,
            notes=("The silhouette may already be clipped; cropping cannot repair it.",),
        )
    box_w = x1 - x0 + 1
    box_h = y1 - y0 + 1
    pad = max(1, int(round(_ratio(query.get("pad_ratio")) * max(box_w, box_h))))
    declared_color = _color(query.get("pad_color"))
    background = Counter(
        image[r][c] for r in range(height) for c in range(width) if not mask[r][c]
    )
    pad_color = declared_color or background.most_common(1)[0][0]
    frame_w = box_w + 2 * pad
    frame_h = box_h + 2 * pad
    frame: Image = [[pad_color for _ in range(frame_w)] for _ in range(frame_h)]
    for r in range(box_h):
        for c in range(box_w):
            frame[pad + r][pad + c] = image[y0 + r][x0 + c]
    rebuilt = [row[pad:pad + box_w] for row in frame[pad:pad + box_h]]
    original = [row[x0:x0 + box_w] for row in image[y0:y0 + box_h]]
    if rebuilt != original:
        raise ValueError("frame reconstruction changed the declared content box")
    kept = sum(1 for r in range(height) for c in range(width) if mask[r][c])
    proposal = propose_anchor({"family": family, "image": image, "mask": mask})
    if proposal.abstained:
        anchor_kind = DEFAULT_ANCHOR[family]
        anchor_x = 0.5
        anchor_y = 1.0 if anchor_kind == "feet" else 0.5
        anchor_source = "deterministic_default"
        confidence = 0.4
    else:
        anchor_kind = proposal.output["anchor"]
        anchor_x = proposal.output["anchor_x"]
        anchor_y = proposal.output["anchor_y"]
        anchor_source = "knn"
        confidence = proposal.confidence
    anchor_px = (
        round(pad + anchor_x * max(box_w - 1, 0), 4),
        round(pad + anchor_y * max(box_h - 1, 0), 4),
    )
    return SpecialistResult(
        SPECIALIST_ID,
        {
            "status": "ready",
            "family": family,
            "image": frame,
            "transform": {
                "crop": {"x": x0, "y": y0, "w": box_w, "h": box_h},
                "pad_px": pad,
                "pad_color": pad_color,
                "frame_size": [frame_w, frame_h],
                "background_color": background.most_common(1)[0][0],
            },
            "anchor": {
                "kind": anchor_kind,
                "normalized": {"x": round(anchor_x, 4), "y": round(anchor_y, 4)},
                "px": {"x": anchor_px[0], "y": anchor_px[1]},
                "source": anchor_source,
            },
            "knn": proposal.output,
            "kept_foreground_pixels": kept,
            "certified": False,
            "source_preserved": True,
        },
        round(min(0.85, float(confidence)), 4),
        False, AuthorityMode.CONSULTATIVE,
        notes=("Geometry only. This is not a rig, a pivot convention or a validated export.",),
    )
