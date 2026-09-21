"""Bounded seam repair for game textures.

The deterministic baseline, the family-local k-NN and the nano model must not
contradict each other. A repair touches a bounded band on both sides of the
wrap plane, reports every change and preserves the source image.
"""

from __future__ import annotations

from typing import Any

from bellium.contracts.schema import AuthorityMode, SpecialistResult
from bellium.knn._image import Image, clamp_rgb, copy_image, shape
from bellium.knn._texture import seam_features
from bellium.knn.tileability import FAMILIES, classify_tileability
from bellium.nano_nn.tile_seam import classify_seam

SPECIALIST_ID = "bellium/hybrid/texture-tile-fixer:v0"
DEFAULT_BAND = 8
MIN_SIZE = 8


def _band(requested: object, width: int, height: int) -> int:
    limit = max(1, min(width, height) // 4)
    if requested is None:
        return min(DEFAULT_BAND, limit)
    if isinstance(requested, bool) or not isinstance(requested, int) or requested < 1:
        raise ValueError("band_px must be a positive integer")
    return min(requested, limit)


def _axis_length(axis: str, width: int, height: int) -> int:
    return width if axis == "x" else height


def _pixel_index(axis: str, primary: int, secondary: int) -> tuple[int, int]:
    # primary walks along the axis, secondary walks across it.
    return (secondary, primary) if axis == "x" else (primary, secondary)


def _feather(image: Image, axis: str, band: int) -> tuple[Image, dict[str, Any]]:
    height, width = shape(image)
    size = _axis_length(axis, width, height)
    across = height if axis == "x" else width
    painted = copy_image(image)
    max_delta = 0
    changed = 0
    for step in range(band):
        # Full closure at the wrap plane, fading to almost nothing inside.
        weight = (band - step) / (2.0 * band)
        for other in range(across):
            left = _pixel_index(axis, step, other)
            right = _pixel_index(axis, size - 1 - step, other)
            for channel in range(3):
                difference = image[left[0]][left[1]][channel] - image[right[0]][right[1]][channel]
                if difference == 0:
                    continue
                shift = difference * weight
                new_left = clamp_rgb(image[left[0]][left[1]][channel] - shift)
                new_right = clamp_rgb(image[right[0]][right[1]][channel] + shift)
                painted[left[0]][left[1]] = _replace(painted[left[0]][left[1]], channel, new_left)
                painted[right[0]][right[1]] = _replace(painted[right[0]][right[1]], channel, new_right)
                max_delta = max(
                    max_delta,
                    abs(new_left - image[left[0]][left[1]][channel]),
                    abs(new_right - image[right[0]][right[1]][channel]),
                )
                changed += 1
    return painted, {
        "axis": axis,
        "band_px": band,
        "changed_channels": changed,
        "max_channel_delta": max_delta,
    }


def _replace(pixel: tuple[int, int, int], channel: int, value: int) -> tuple[int, int, int]:
    channels = list(pixel)
    channels[channel] = value
    return channels[0], channels[1], channels[2]


def fix_texture_seams(query: dict[str, Any]) -> SpecialistResult:
    image = query.get("image")
    if image is None:
        raise ValueError("query needs image")
    height, width = shape(image)
    family = str(query.get("family") or "").strip()
    if family not in FAMILIES:
        return SpecialistResult(
            SPECIALIST_ID,
            {"status": "abstain", "reason": "unknown_family", "family": family,
             "image": None, "corrections": []},
            0.0, True, AuthorityMode.CONSULTATIVE,
            notes=("Unknown texture family; nothing is repaired.",),
        )
    if min(height, width) < MIN_SIZE:
        return SpecialistResult(
            SPECIALIST_ID,
            {"status": "abstain", "reason": "image_too_small", "family": family,
             "image": None, "corrections": []},
            0.0, True, AuthorityMode.CONSULTATIVE,
        )
    band = _band(query.get("band_px"), width, height)
    verdict = classify_tileability({"family": family, "image": image})
    if verdict.abstained:
        return SpecialistResult(
            SPECIALIST_ID,
            {"status": "abstain", "reason": "tileability_not_confirmed", "family": family,
             "image": None, "corrections": [], "knn": verdict.output},
            0.0, True, AuthorityMode.CONSULTATIVE,
            notes=("Wrap continuity was not established; no seam is touched.",),
        )
    measured = seam_features(image)
    axes: dict[str, Any] = {}
    planned: list[str] = []
    for axis in ("x", "y"):
        axis_state = verdict.output["axes"][axis]
        if axis_state["status"] != "suggest":
            axes[axis] = {"decision": "skip",
                          "reason": axis_state.get("reason") or "unconfirmed"}
            continue
        if axis_state["verdict"] == "tileable":
            axes[axis] = {"decision": "skip", "reason": "already_continuous",
                          "gap": round(measured[axis]["gap"], 4)}
            continue
        nano = classify_seam(measured[axis])
        if not nano.abstained and nano.output["verdict"] == "continuous":
            axes[axis] = {"decision": "abstain", "reason": "nano_disagrees",
                          "gap": round(measured[axis]["gap"], 4),
                          "nano": nano.output, "baseline": axis_state["baseline"]}
            continue
        axes[axis] = {
            "decision": "repair",
            "reason": "confirmed_seam",
            "gap": round(measured[axis]["gap"], 4),
            "baseline": axis_state["baseline"],
            "nano": nano.output,
            "nano_abstained": nano.abstained,
        }
        planned.append(axis)
    if not planned:
        return SpecialistResult(
            SPECIALIST_ID,
            {"status": "abstain", "reason": "no_confirmed_seam", "family": family,
             "image": None, "corrections": [], "axes": axes, "knn": verdict.output},
            0.0, True, AuthorityMode.CONSULTATIVE,
            notes=("Every axis was skipped or contradicted; the texture is untouched.",),
        )
    working = copy_image(image)
    corrections = []
    for axis in planned:
        working, report = _feather(working, axis, band)
        after = seam_features(working)[axis]
        axes[axis]["gap_after"] = round(after["gap"], 4)
        axes[axis]["improved"] = after["gap"] < measured[axis]["gap"]
        corrections.append(report)
    confidence = min(
        [axes[axis].get("nano", {}).get("confidence") or verdict.confidence for axis in planned]
    )
    return SpecialistResult(
        SPECIALIST_ID,
        {
            "status": "suggest",
            "family": family,
            "image": working,
            "corrections": corrections,
            "axes": axes,
            "knn": verdict.output,
            "certified": False,
            "source_preserved": True,
        },
        round(min(0.85, float(confidence)), 4),
        False, AuthorityMode.CONSULTATIVE,
        notes=("Candidate only. Visual quality of the blend is not established by this test.",),
    )
