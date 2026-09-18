"""Experimental local selection between patch copying and axis interpolation.

Opt-in research API. The default patch filler and all learned weights are unchanged.
"""
from __future__ import annotations

from bellium.contracts import AuthorityMode, SpecialistResult
from bellium.knn._image import Image, Mask, clamp_rgb, copy_image, validate_mask
from bellium.knn.patch_inpaint import (
    K, SEARCH, MAX_PROBE_MAE_255, MAX_PROBE_PIXELS, MIN_COMPLETE_PROBES,
    _inpaint_candidate, _probe_regions, _validate_options,
)

SPECIALIST_ID = "bellium/hybrid/adaptive-inpaint:experimental-v1"
METHODS = ("patch_knn", "axis_linear")


def _axis_linear(image: Image, mask: Mask) -> Image | None:
    """Interpolate only between originally known endpoints; never read a hole."""
    height, width = len(image), len(image[0])
    output = copy_image(image)
    for r in range(height):
        for c in range(width):
            if not mask[r][c]:
                continue
            estimates = []
            for dr, dc in ((0, 1), (1, 0)):
                before = after = 1
                while 0 <= r-before*dr < height and 0 <= c-before*dc < width and mask[r-before*dr][c-before*dc]:
                    before += 1
                while 0 <= r+after*dr < height and 0 <= c+after*dc < width and mask[r+after*dr][c+after*dc]:
                    after += 1
                a, b = (r-before*dr, c-before*dc), (r+after*dr, c+after*dc)
                if not (0 <= a[0] < height and 0 <= a[1] < width and 0 <= b[0] < height and 0 <= b[1] < width):
                    continue
                span = before + after
                color = tuple((image[a[0]][a[1]][i]*after + image[b[0]][b[1]][i]*before)/span for i in range(3))
                estimates.append((1.0/(span*span), color))
            if not estimates:
                return None
            weight = sum(w for w, _ in estimates)
            output[r][c] = tuple(clamp_rgb(sum(w*p[i] for w, p in estimates)/weight) for i in range(3))
    return output


def inpaint(
    image: Image, mask: Mask, *, patch_size: int = 3, search_radius: int = SEARCH,
    k_neighbors: int = K,
) -> SpecialistResult:
    validate_mask(image, mask)
    _validate_options(patch_size, search_radius, k_neighbors)
    count = sum(sum(row) for row in mask)
    if count > MAX_PROBE_PIXELS:
        return SpecialistResult(SPECIALIST_ID, {"reason": "quality_probe_budget_exceeded"},
                                None, True, AuthorityMode.CONSULTATIVE)
    patch = _inpaint_candidate(image, mask, patch_size=patch_size,
                              search_radius=search_radius, k_neighbors=k_neighbors)
    if count == 0:
        return SpecialistResult(SPECIALIST_ID, patch.output, 1.0, False, AuthorityMode.CONSULTATIVE)
    # Public size/domain vetoes still apply to every method.
    if patch.output.get("reason") == "hole_too_large":
        return SpecialistResult(SPECIALIST_ID, patch.output, None, True, AuthorityMode.CONSULTATIVE)
    candidates = {
        "patch_knn": patch.output.get("image") if patch.output.get("status") == "complete" else None,
        "axis_linear": _axis_linear(image, mask),
    }
    regions = _probe_regions(mask)
    scores = {name: [] for name in METHODS}
    for region in regions:
        probe_mask = [row[:] for row in mask]
        probe_image = copy_image(image)
        for r, row in enumerate(mask):
            for c, missing in enumerate(row):
                if missing:
                    probe_image[r][c] = (0, 0, 0)
        for r, c in region:
            probe_mask[r][c] = 1
            probe_image[r][c] = (0, 0, 0)
        trial = _inpaint_candidate(probe_image, probe_mask, patch_size=patch_size,
                                   search_radius=search_radius, k_neighbors=k_neighbors, _probe=True)
        predictions = {
            "patch_knn": trial.output.get("image") if trial.output.get("status") == "complete" else None,
            "axis_linear": _axis_linear(probe_image, probe_mask),
        }
        for name, prediction in predictions.items():
            if prediction is None:
                continue
            error = sum(abs(image[r][c][i]-prediction[r][c][i])
                        for r, c in region for i in range(3)) / (3*len(region))
            scores[name].append(error)
    records = {}
    eligible = []
    for name in METHODS:
        errors = scores[name]
        maximum = max(errors) if errors else None
        enough = len(errors) == len(regions) and len(errors) >= MIN_COMPLETE_PROBES
        passes = candidates[name] is not None and enough and maximum <= MAX_PROBE_MAE_255
        records[name] = {"complete_probes": len(errors), "probe_mae_255": errors,
                         "max_probe_mae_255": maximum, "passed": passes}
        if passes:
            eligible.append((maximum, METHODS.index(name), name))
    selected = min(eligible)[2] if eligible else None
    output = {"image": candidates[selected] if selected else patch.output.get("image", copy_image(image)),
              "filled": count if selected else 0, "selected_method": selected,
              "status": "complete" if selected else "uncertain",
              "quality": {"version": "method-selection-v1", "methods": records,
                          "attempted_probes": len(regions), "limit_mae_255": MAX_PROBE_MAE_255,
                          "passed": selected is not None, "calibrated_probability": False}}
    if selected is None:
        output["reason"] = "no_validated_method"
    return SpecialistResult(
        SPECIALIST_ID, output, None, selected is None, AuthorityMode.CONSULTATIVE,
        notes=("Experimental method selection on known context, not a probability or permission to act.",),
    )
