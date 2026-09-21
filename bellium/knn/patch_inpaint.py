"""Patch k-NN fill for small holes. Consultative, CPU, no private memory."""

from __future__ import annotations

from dataclasses import replace
from heapq import nsmallest

from bellium.contracts.schema import AuthorityMode, SpecialistResult
from bellium.knn._image import Image, Mask, clamp_rgb, copy_image, shape, validate_mask

SPECIALIST_ID = "bellium/knn/patch-inpaint:v0"
MAX_AREA_RATIO = 0.12
PATCH = 1  # 3x3
SEARCH = 6
K = 5
QUALITY_GUARD_VERSION = "local-probes-v1"
MAX_PROBE_MAE_255 = 12.0
MIN_COMPLETE_PROBES = 2
MAX_PROBE_PIXELS = 256
PROBE_WINDOW = 8
MAX_CONTEXT_MAE = 18.0
PROBE_RING = 2
PROBE_CANDIDATE_LIMIT = 120
MAX_PROBES = 4


def _hole_stats(mask: Mask) -> tuple[int, int, int]:
    count = 0
    min_r = min_c = 10**9
    max_r = max_c = -1
    for r, row in enumerate(mask):
        for c, value in enumerate(row):
            if value:
                count += 1
                min_r, max_r = min(min_r, r), max(max_r, r)
                min_c, max_c = min(min_c, c), max(max_c, c)
    span = 0 if count == 0 else max(max_r - min_r + 1, max_c - min_c + 1)
    return count, span, len(mask) * len(mask[0])


def _patch_ok(mask: Mask, r: int, c: int, radius: int) -> bool:
    height, width = len(mask), len(mask[0])
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            rr, cc = r + dr, c + dc
            if rr < 0 or cc < 0 or rr >= height or cc >= width:
                return False
            if mask[rr][cc]:
                return False
    return True


def _flatten_patch(image: Image, r: int, c: int, radius: int) -> list[float]:
    vec: list[float] = []
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            pixel = image[r + dr][c + dc]
            vec.extend(float(channel) / 255.0 for channel in pixel)
    return vec


def _ssd(left: list[float], right: list[float]) -> float:
    return sum((a - b) ** 2 for a, b in zip(left, right))


def _flatten_patch_raw(image: Image, r: int, c: int, radius: int) -> list[float]:
    """Raw 0-255 channels, matching the scale used by the distance functions."""
    vec: list[float] = []
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            vec.extend(float(channel) for channel in image[r + dr][c + dc])
    return vec


def _known_patch_distance(
    query: Image, remaining: Mask, r: int, c: int,
    source: Image, sr: int, sc: int, radius: int = PATCH,
) -> float:
    height, width = len(query), len(query[0])
    total = 0.0
    known = 0
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            rr, cc = r + dr, c + dc
            if 0 <= rr < height and 0 <= cc < width and not remaining[rr][cc]:
                total += sum((a - b) ** 2 for a, b in zip(query[rr][cc], source[sr + dr][sc + dc]))
                known += 3
    return total / (known * 255.0 ** 2) if known else float("inf")


def _patch_distance(
    query: Image, remaining: Mask, r: int, c: int, flat_source: list[float], radius: int,
) -> float:
    """Same distance as `_known_patch_distance`, reading a precomputed source patch.

    Flattening the source patch once per call instead of once per (pixel, candidate)
    pair removes the dominant cost of the bounded probe fills.
    """
    height, width = len(query), len(query[0])
    total = 0.0
    known = 0
    index = 0
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            rr, cc = r + dr, c + dc
            base = index * 3
            if 0 <= rr < height and 0 <= cc < width and not remaining[rr][cc]:
                pixel = query[rr][cc]
                total += (pixel[0] - flat_source[base]) ** 2
                total += (pixel[1] - flat_source[base + 1]) ** 2
                total += (pixel[2] - flat_source[base + 2]) ** 2
                known += 3
            index += 1
    return total / (known * 255.0 ** 2) if known else float("inf")


def _validate_options(patch_size: int, search_radius: int, k_neighbors: int) -> None:
    if isinstance(patch_size, bool) or not isinstance(patch_size, int) or patch_size < 3 or patch_size % 2 == 0:
        raise ValueError("patch_size must be an odd integer of at least 3")
    if isinstance(search_radius, bool) or not isinstance(search_radius, int) or search_radius < 1:
        raise ValueError("search_radius must be a positive integer")
    if isinstance(k_neighbors, bool) or not isinstance(k_neighbors, int) or k_neighbors < 2:
        raise ValueError("k_neighbors must be an integer of at least 2")


def _inpaint_candidate(
    image: Image, mask: Mask, *, patch_size: int = 3, search_radius: int = SEARCH,
    k_neighbors: int = K, _probe: bool = False,
) -> SpecialistResult:
    validate_mask(image, mask)
    _validate_options(patch_size, search_radius, k_neighbors)
    radius = patch_size // 2
    height, width = shape(image)
    hole_count, span, total = _hole_stats(mask)
    if hole_count == 0:
        return SpecialistResult(
            SPECIALIST_ID, {"image": copy_image(image), "filled": 0},
            1.0, False, AuthorityMode.CONSULTATIVE, notes=("Nothing to fill.",),
        )
    area_ratio = hole_count / total
    if not _probe and (area_ratio > MAX_AREA_RATIO or span > max(height, width) * 0.45):
        return SpecialistResult(
            SPECIALIST_ID,
            {"reason": "hole_too_large", "area_ratio": round(area_ratio, 4)},
            0.0, True, AuthorityMode.CONSULTATIVE,
            notes=("Escalate to a heavier inpainting model.",),
        )

    filled = copy_image(image)
    remaining_mask = [row[:] for row in mask]
    # A masked pixel only ever looks `search_radius` away for a source patch, and
    # every masked pixel lies inside the hole bounding box. Restricting the source
    # scan to that box plus the radius is therefore equivalent, and turns an
    # image-sized scan into a bounded local one.
    hole_rows = [r for r in range(height) for c in range(width) if mask[r][c]]
    hole_cols = [c for r in range(height) for c in range(width) if mask[r][c]]
    row_low, row_high = max(radius, min(hole_rows) - search_radius), min(height - radius, max(hole_rows) + search_radius + 1)
    col_low, col_high = max(radius, min(hole_cols) - search_radius), min(width - radius, max(hole_cols) + search_radius + 1)
    sources = {
        (r, c) for r in range(row_low, row_high) for c in range(col_low, col_high)
        if _patch_ok(mask, r, c, radius)
    }
    if len(sources) < k_neighbors:
        return SpecialistResult(
            SPECIALIST_ID, {"reason": "not_enough_source_patches"},
            0.0, True, AuthorityMode.CONSULTATIVE,
            notes=("Too few unmasked patches to copy from.",),
        )

    flat_sources = {position: _flatten_patch_raw(image, position[0], position[1], radius)
                    for position in sources}

    remaining = {(r, c) for r in range(height) for c in range(width) if mask[r][c]}
    distances = []
    while remaining:
        updates = []
        for r, c in sorted(remaining):
            if not any(
                0 <= r + dr < height and 0 <= c + dc < width and not remaining_mask[r + dr][c + dc]
                for dr in range(-radius, radius + 1) for dc in range(-radius, radius + 1)
            ):
                continue
            candidates = (
                (sr, sc) for sr in range(max(radius, r - search_radius), min(height - radius, r + search_radius + 1))
                for sc in range(max(radius, c - search_radius), min(width - radius, c + search_radius + 1))
                if (sr, sc) in sources
            )
            nearest = nsmallest(k_neighbors, (
                (_patch_distance(filled, remaining_mask, r, c, flat_sources[(sr, sc)], radius), (sr, sc))
                for sr, sc in candidates
            ))
            if len(nearest) < k_neighbors:
                continue
            acc = [0.0, 0.0, 0.0]
            weight_sum = 0.0
            for dist, (sr, sc) in nearest:
                weight = 1.0 / (dist + 1e-6)
                pixel = image[sr][sc]
                for i in range(3):
                    acc[i] += weight * pixel[i]
                weight_sum += weight
            updates.append((r, c, tuple(clamp_rgb(acc[i] / weight_sum) for i in range(3))))
            distances.append(nearest[0][0])
        if not updates:
            break
        # Apply a complete frontier together, so traversal order does not change context.
        for r, c, pixel in updates:
            filled[r][c] = pixel
            remaining_mask[r][c] = 0
            remaining.remove((r, c))

    filled_pixels = hole_count - len(remaining)
    confidence = 0.0 if remaining else max(0.0, min(0.95, 1.0 - max(distances, default=0.0)))
    abstained = bool(remaining) or confidence < 0.55
    return SpecialistResult(
        SPECIALIST_ID,
        {
            "image": filled,
            "filled": filled_pixels,
            "hole_count": hole_count,
            "weak": len(remaining),
            "remaining_mask": remaining_mask,
            "status": "partial" if remaining else "complete",
        },
        round(confidence, 4),
        abstained,
        AuthorityMode.CONSULTATIVE,
        notes=("Patch k-NN is for small, simple holes only.",),
    )


def _hole_cells(mask: Mask) -> list[tuple[int, int]]:
    return [(r, c) for r in range(len(mask)) for c in range(len(mask[0])) if mask[r][c]]


def _probe_ring(mask: Mask) -> list[tuple[int, int]]:
    """Absolute coordinates of the known cells surrounding the hole."""
    height, width = len(mask), len(mask[0])
    hole = set(_hole_cells(mask))
    ring = []
    for r, c in hole:
        for dr in range(-PROBE_RING, PROBE_RING + 1):
            for dc in range(-PROBE_RING, PROBE_RING + 1):
                cell = (r + dr, c + dc)
                if cell in hole or cell in ring:
                    continue
                if 0 <= cell[0] < height and 0 <= cell[1] < width:
                    ring.append(cell)
    return ring


def _context_mae(image: Image, mask: Mask, region: list[tuple[int, int]],
                 ring: list[tuple[int, int]]) -> float | None:
    """Mean absolute difference between the hole's ring and the probe's ring.

    Both rings sample identical offsets, so only local appearance is compared.
    Cells hidden by either mask are skipped; at least twelve samples are required.
    """
    hole = set(_hole_cells(mask))
    probe = set(region)
    height, width = len(mask), len(mask[0])
    delta_row = min(r for r, _ in region) - min(r for r, _ in hole)
    delta_col = min(c for _, c in region) - min(c for _, c in hole)
    pairs = []
    for row, col in ring:
        target = (row + delta_row, col + delta_col)
        if target in probe or target in hole:
            continue
        if not (0 <= target[0] < height and 0 <= target[1] < width):
            continue
        if mask[target[0]][target[1]]:
            continue
        pairs.append((image[row][col], image[target[0]][target[1]]))
    if len(pairs) < 12:
        return None
    budget = len(pairs) * 3 * MAX_CONTEXT_MAE
    total = 0.0
    count = 0
    for source, target in pairs:
        for channel in range(3):
            total += abs(source[channel] - target[channel])
            count += 1
        if total > budget:
            return MAX_CONTEXT_MAE + 1.0
    return total / count


def _probe_candidates(mask: Mask) -> list[list[tuple[int, int]]]:
    height, width = len(mask), len(mask[0])
    hole = _hole_cells(mask)
    if not hole:
        return []
    row_span = max(r for r, _ in hole) - min(r for r, _ in hole) + 1
    col_span = max(c for _, c in hole) - min(c for _, c in hole) + 1
    candidates = []
    offsets = [(dr, dc)
               for dr in range(-(row_span + PROBE_WINDOW), row_span + PROBE_WINDOW + 1)
               for dc in range(-(col_span + PROBE_WINDOW), col_span + PROBE_WINDOW + 1)
               if not (dr == 0 and dc == 0)]
    # Nearest displacements first, so the bounded budget spends its samples on the
    # sites most likely to share the hole's local statistics.
    offsets.sort(key=lambda item: (item[0] * item[0] + item[1] * item[1], item))
    for dr, dc in offsets:
        shifted = [(r + dr, c + dc) for r, c in hole]
        if all(0 <= r < height and 0 <= c < width and not mask[r][c] for r, c in shifted):
            candidates.append(shifted)
            if len(candidates) == PROBE_CANDIDATE_LIMIT:
                break
    return candidates


def _probe_regions(image: Image, mask: Mask, *, max_probes: int = MAX_PROBES) -> tuple[list[list[tuple[int, int]]], int]:
    """Pick probe sites whose known surroundings look most like the hole's."""
    candidates = _probe_candidates(mask)
    considered = len(candidates)
    ring = _probe_ring(mask)
    scored = []
    for region in candidates:
        score = _context_mae(image, mask, region, ring)
        if score is None or score > MAX_CONTEXT_MAE:
            continue
        scored.append((score, region))
    scored.sort(key=lambda item: item[0])
    chosen: list[list[tuple[int, int]]] = []
    used: set[tuple[int, int]] = set(_hole_cells(mask))
    for _score, region in scored:
        if used.intersection(region):
            continue
        chosen.append(region)
        used.update(region)
        if len(chosen) == max_probes:
            break
    return chosen, considered


def _check_local_reconstruction(
    image: Image, mask: Mask, candidate_image: Image, *, patch_size: int,
    search_radius: int, k_neighbors: int, max_probes: int = MAX_PROBES,
) -> dict:
    """Replay each probe with a single masked block, on the candidate fill.

    Masking the original hole as well would make the probe harder than the real
    task. The candidate supplies the hole content, so no true hidden pixel is read.
    """
    errors = []
    regions, considered = _probe_regions(image, mask, max_probes=max_probes)
    for region in regions:
        probe_image = copy_image(candidate_image)
        probe_mask = [[0] * len(mask[0]) for _ in mask]
        for r, c in region:
            probe_mask[r][c] = 1
        probe = _inpaint_candidate(
            probe_image, probe_mask, patch_size=patch_size,
            search_radius=search_radius, k_neighbors=k_neighbors, _probe=True,
        )
        prediction = probe.output.get("image")
        if prediction is None or probe.output.get("status") != "complete":
            continue
        error = sum(abs(image[r][c][channel] - prediction[r][c][channel])
                    for r, c in region for channel in range(3)) / (3 * len(region))
        errors.append(error)
    enough = len(errors) >= MIN_COMPLETE_PROBES
    maximum = max(errors) if errors else None
    return {
        "version": QUALITY_GUARD_VERSION,
        "complete_probes": len(errors),
        "required_probes": MIN_COMPLETE_PROBES,
        "probe_window": PROBE_WINDOW,
        "max_context_mae_255": MAX_CONTEXT_MAE,
        "probe_mae_255": [round(error, 6) for error in errors],
        "max_probe_mae_255": round(maximum, 6) if maximum is not None else None,
        "limit_mae_255": MAX_PROBE_MAE_255,
        "candidate_sites": considered,
        "selected_sites": len(regions),
        "passed": enough and maximum is not None and maximum <= MAX_PROBE_MAE_255,
        "reason": ("unpredictable_context" if considered and len(regions) < MIN_COMPLETE_PROBES
                   else "insufficient_quality_probes" if not enough
                   else "local_reconstruction_error" if maximum is not None and maximum > MAX_PROBE_MAE_255
                   else None),
        "calibrated_probability": False,
    }


def inpaint(
    image: Image, mask: Mask, *, patch_size: int = 3, search_radius: int = SEARCH,
    k_neighbors: int = K,
) -> SpecialistResult:
    """Propose a small-hole fill only when bounded known-context replays succeed.

    Nearby probe errors are evidence, not a guarantee about unseen pixels.
    Confidence is None for filled candidates: the former support score is not a
    calibrated probability and is now exposed under output.support_score only.
    """
    validate_mask(image, mask)
    _validate_options(patch_size, search_radius, k_neighbors)
    hole_count, span, total = _hole_stats(mask)
    height, width = len(image), len(image[0])
    if (hole_count <= MAX_PROBE_PIXELS or hole_count / total > MAX_AREA_RATIO
            or span > max(height, width) * 0.45):
        candidate = _inpaint_candidate(
            image, mask, patch_size=patch_size,
            search_radius=search_radius, k_neighbors=k_neighbors,
        )
    else:
        return SpecialistResult(
            SPECIALIST_ID,
            {"reason": "quality_probe_budget_exceeded", "hole_count": hole_count,
             "quality": {"version": QUALITY_GUARD_VERSION, "max_probe_pixels": MAX_PROBE_PIXELS,
                         "passed": False, "calibrated_probability": False}},
            None, True, AuthorityMode.CONSULTATIVE,
        )
    if hole_count == 0 or candidate.abstained:
        return candidate
    quality = _check_local_reconstruction(
        image, mask, candidate.output["image"], patch_size=patch_size,
        search_radius=search_radius, k_neighbors=k_neighbors,
    )
    output = {**candidate.output, "quality": quality, "support_score": candidate.confidence}
    if not quality["passed"]:
        output.update({"status": "uncertain", "reason": quality["reason"]})
    return replace(
        candidate, output=output, confidence=None, abstained=not quality["passed"],
        notes=(*candidate.notes, "Known-context reconstruction probes are not calibrated probabilities."),
    )
