from copy import deepcopy
import random

from bellium.specialists.adaptive_inpaint import _axis_linear, inpaint


def _mask(size=24, side=5):
    return [[int(9 <= r < 9+side and 9 <= c < 9+side) for c in range(size)] for r in range(size)]


def test_axis_interpolation_recovers_a_plane_without_touching_known_pixels():
    original = [[(20+3*r+2*c, 30+r+c, 80) for c in range(24)] for r in range(24)]
    mask = _mask()
    corrupted = [[(0, 0, 0) if mask[r][c] else original[r][c] for c in range(24)] for r in range(24)]
    assert _axis_linear(corrupted, mask) == original


def test_selection_accepts_known_predictable_context_and_keeps_inputs():
    image = [[(20+3*r+2*c, 30+r+c, 80) for c in range(24)] for r in range(24)]
    mask = _mask()
    before = deepcopy(image)
    result = inpaint(image, mask)
    assert not result.abstained
    assert result.output["selected_method"] == "axis_linear"
    assert result.output["image"] == image
    assert image == before and result.confidence is None


def test_texture_without_local_predictability_still_abstains():
    rng = random.Random(719)
    image = [[tuple(rng.randrange(256) for _ in range(3)) for _ in range(24)] for _ in range(24)]
    result = inpaint(image, _mask())
    assert result.abstained and result.output["selected_method"] is None


def test_selection_does_not_look_at_masked_values():
    image = [[(100, 120, 140)]*24 for _ in range(24)]
    mask = _mask()
    changed = [[(255, 0, 255) if mask[r][c] else image[r][c] for c in range(24)] for r in range(24)]
    assert inpaint(image, mask) == inpaint(changed, mask)
