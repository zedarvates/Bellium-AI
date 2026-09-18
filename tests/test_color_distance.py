from collections import Counter
import math
import random

import pytest

from bellium.knn.color_cutout import _distance, cutout, MAX_BORDER_COLORS


def test_compacted_color_search_preserves_exact_k_nearest_distance():
    rng = random.Random(37)
    for _ in range(30):
        palette = [tuple(rng.randrange(256) for _ in range(3)) for _ in range(8)]
        samples = rng.choices(palette, k=100)
        query = tuple(rng.randrange(256) for _ in range(3))
        original = sorted(math.dist(query, p) for p in samples)[:5]
        assert _distance(query, Counter(samples)) == pytest.approx(sum(original) / 5)


def test_over_budget_color_space_abstains():
    size = MAX_BORDER_COLORS // 4
    image = [[(96 + (r % 16) * 3, 96 + (c % 16) * 3, 96 + (r // 16) * 8 + c // 16)
              for c in range(size)] for r in range(size)]
    result = cutout(image)
    assert result.abstained
    assert result.output["reason"] == "too_many_border_colors"
