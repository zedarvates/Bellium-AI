from copy import deepcopy
import random

from bellium.knn.patch_inpaint import inpaint


def _hole(image, side=3):
    size = len(image)
    mask = [[int(size//2-side//2 <= r < size//2-side//2+side and
                 size//2-side//2 <= c < size//2-side//2+side)
             for c in range(size)] for r in range(size)]
    corrupted = [[(0, 0, 0) if mask[r][c] else image[r][c] for c in range(size)] for r in range(size)]
    return corrupted, mask


def test_unpredictable_visible_texture_abstains():
    rng = random.Random(831)
    image = [[tuple(rng.randrange(256) for _ in range(3)) for _ in range(32)] for _ in range(32)]
    corrupted, mask = _hole(image)
    result = inpaint(corrupted, mask)
    assert result.abstained
    assert result.output["reason"] == "local_reconstruction_error"
    assert result.output["quality"]["max_probe_mae_255"] > 12
    assert result.confidence is None


def test_uniform_context_has_measurable_support_without_fake_probability():
    corrupted, mask = _hole([[(40, 80, 120)] * 24 for _ in range(24)], side=5)
    original, original_mask = deepcopy(corrupted), deepcopy(mask)
    result = inpaint(corrupted, mask)
    assert not result.abstained
    assert result.output["quality"]["complete_probes"] >= 2
    assert result.output["quality"]["max_probe_mae_255"] == 0
    assert result.confidence is None
    assert result.output["quality"]["calibrated_probability"] is False
    assert corrupted == original and mask == original_mask


def test_probe_estimate_does_not_read_the_true_missing_pixels():
    corrupted, mask = _hole([[(40, 80, 120)] * 24 for _ in range(24)])
    different = deepcopy(corrupted)
    for r, row in enumerate(mask):
        for c, missing in enumerate(row):
            if missing:
                different[r][c] = (255, 0, 255)
    left, right = inpaint(corrupted, mask), inpaint(different, mask)
    assert left.output["quality"] == right.output["quality"]
    assert left.output["image"] == right.output["image"]


def test_probe_budget_is_explicit_instead_of_accepting_unchecked_result():
    corrupted, mask = _hole([[(40, 80, 120)] * 64 for _ in range(64)], side=17)
    result = inpaint(corrupted, mask)
    assert result.abstained
    assert result.output["reason"] == "quality_probe_budget_exceeded"


def test_noop_remains_identity_without_probes():
    image = [[(20, 20, 20)] * 8 for _ in range(8)]
    result = inpaint(image, [[0]*8 for _ in range(8)])
    assert not result.abstained and result.output["filled"] == 0
    assert result.output["image"] == image
