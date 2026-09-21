"""The guard must accept reconstructible structure and refuse noise, cheaply."""
import random
import time

from bellium.knn.patch_inpaint import inpaint


def _hole(size, side):
    base = size // 2 - side // 2
    return [[int(base <= r < base + side and base <= c < base + side) for c in range(size)]
            for r in range(size)]


def _apply(image, mask, value=(0, 0, 0)):
    return [[value if mask[r][c] else image[r][c] for c in range(len(mask[0]))]
            for r in range(len(mask))]


def _run(image, side=5):
    mask = _hole(len(image), side)
    return inpaint(_apply(image, mask), mask)


def test_smooth_structure_is_accepted_with_low_error():
    image = [[(40 + r, 60 + c, 120) for c in range(48)] for r in range(48)]
    result = _run(image)
    assert not result.abstained
    returned = result.output["image"]
    error = sum(abs(returned[r][c][i] - image[r][c][i])
                for r in range(48) for c in range(48)
                if _hole(48, 5)[r][c] for i in range(3)) / (3 * 25)
    assert error <= 2


def test_periodic_texture_is_accepted_when_the_period_matches_the_probe_window():
    image = [[(220, 210, 200) if (c // 6) % 2 else (40, 50, 60) for c in range(48)]
             for r in range(48)]
    result = _run(image, side=3)
    assert not result.abstained


def test_noise_is_refused_by_the_context_test():
    rng = random.Random(99)
    image = [[tuple(rng.randrange(256) for _ in range(3)) for _ in range(48)] for _ in range(48)]
    result = _run(image)
    assert result.abstained
    assert result.output["reason"] in {"unpredictable_context", "local_reconstruction_error"}


def test_guard_stays_within_a_small_time_budget_on_a_moderate_hole():
    image = [[(40 + r % 7, 60 + c % 5, 120) for c in range(64)] for r in range(64)]
    mask = _hole(64, 7)
    start = time.perf_counter()
    result = inpaint(_apply(image, mask), mask)
    elapsed = time.perf_counter() - start
    assert elapsed < 2.5, elapsed
    assert not result.abstained


def test_probe_sites_reject_locations_with_a_different_context():
    from bellium.knn.patch_inpaint import MAX_CONTEXT_MAE, _context_mae, _probe_candidates, _probe_ring
    size = 40
    image = [[(30, 30, 30) for _ in range(size)] for _ in range(size)]
    for r in range(size):
        for c in range(20, size):
            image[r][c] = (200, 40, 40)
    mask = [[0] * size for _ in range(size)]
    for r in range(15, 20):
        for c in range(15, 20):
            mask[r][c] = 1
    ring = _probe_ring(mask)
    across = [region for region in _probe_candidates(mask)
              if min(cell[1] for cell in region) > 20]
    assert across, "expected candidate sites on the other side of the colour edge"
    scores = [_context_mae(image, mask, region, ring) for region in across[:5]]
    assert all(score is not None and score > MAX_CONTEXT_MAE for score in scores), scores
    # A site inside the same flat region matches perfectly.
    same = [region for region in _probe_candidates(mask)
            if max(cell[1] for cell in region) < 15 and min(cell[0] for cell in region) > 15]
    assert same
    # The hole's ring already straddles the colour edge, so any translation scores
    # above zero. What matters is that moving across the edge is clearly worse.
    same_score = _context_mae(image, mask, same[0], ring)
    assert same_score is not None and same_score < max(scores)
