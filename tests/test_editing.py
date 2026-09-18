from copy import deepcopy

import pytest

from bellium.editing import (
    EditStack,
    EditStep,
    apply_filter,
    apply_mask,
    brightness_contrast,
    brush_mask,
    ellipse_mask,
    feather,
    grayscale,
    invert,
    polygon_mask,
    rect_mask,
    saturation,
    sepia,
    threshold,
    tint,
)
from bellium.specialists.editing import edit_image, erase_region


def _image(size=4, color=(120, 60, 30)):
    return [[color for _ in range(size)] for _ in range(size)]


def test_grayscale_uses_the_documented_luma_weights() -> None:
    assert grayscale([[(255, 0, 0)]])[0][0] == (54, 54, 54)
    assert grayscale([[(0, 0, 255)]])[0][0] == (18, 18, 18)
    assert grayscale([[(255, 0, 0)]], weights="average")[0][0] == (85, 85, 85)


def test_strength_blends_between_source_and_filter() -> None:
    source = [[(255, 0, 0)]]
    half = grayscale(source, strength=0.5)[0][0]
    assert half == (154, 27, 27)
    assert grayscale(source, strength=0.0)[0][0] == (255, 0, 0)
    assert grayscale(source, strength=1.0)[0][0] == (54, 54, 54)


def test_counts_the_primary_filters() -> None:
    assert invert([[(0, 128, 255)]])[0][0] == (255, 127, 0)
    assert sepia([[(255, 255, 255)]])[0][0] == (255, 255, 239)
    assert threshold([[(255, 255, 255)]], level=0.5)[0][0] == (255, 255, 255)
    assert threshold([[(10, 10, 10)]], level=0.5)[0][0] == (0, 0, 0)
    assert brightness_contrast([[(100, 100, 100)]], brightness=0.2)[0][0] == (151, 151, 151)
    # The grey level is the documented luma, not the channel mean.
    assert saturation([[(200, 100, 50)]], factor=0.0)[0][0] == (118, 118, 118)
    assert tint([[(100, 100, 100)]], factors=(2.0, 1.0, 0.5))[0][0] == (200, 100, 50)


def test_filters_never_write_to_the_source() -> None:
    source = _image()
    before = deepcopy(source)
    for name in ("grayscale", "sepia", "invert", "threshold"):
        apply_filter(source, name)
    assert source == before


def test_unknown_filter_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown filter"):
        apply_filter(_image(), "oil-paint")


def test_declared_parameters_are_validated() -> None:
    for call in (
        lambda: grayscale(_image(), strength=1.5),
        lambda: grayscale(_image(), weights="cyan"),
        lambda: threshold(_image(), level=2.0),
        lambda: tint(_image(), factors=(1.0, 1.0)),
    ):
        blocked = False
        try:
            call()
        except ValueError:
            blocked = True
        assert blocked


def test_shapes_and_brush_strokes_build_masks() -> None:
    assert sum(sum(row) for row in rect_mask((3, 3), (0, 0, 2, 2))) == 4.0
    assert sum(sum(row) for row in ellipse_mask((5, 5), (2, 2), (1, 1))) == 5.0
    assert sum(sum(row) for row in polygon_mask((5, 5), [(0, 0), (4, 0), (2, 4)])) == 10.0
    stroke = brush_mask((8, 8), [(1, 1), (6, 6)], radius=1.0)
    assert stroke[1][1] == 1.0 and stroke[6][6] == 1.0 and stroke[4][4] == 1.0
    assert stroke[0][7] == 0.0


def test_feather_softens_edges_without_exceeding_the_mask() -> None:
    mask = rect_mask((6, 6), (2, 2, 2, 2))
    soft = feather(mask, radius=1)
    assert 0.0 < soft[1][2] < 1.0
    assert soft[2][2] > soft[1][2] > soft[0][2]
    assert soft[2][2] < 1.0
    assert all(0.0 <= value <= 1.0 for row in soft for value in row)


def test_masked_edit_preserves_uncovered_pixels() -> None:
    source = _image(6, (200, 100, 50))
    mask = feather(rect_mask((6, 6), (2, 2, 2, 2)), radius=1)
    edited = apply_mask(source, invert(source), mask)
    assert edited[0][0] == source[0][0]
    assert edited[5][5] == source[5][5]
    assert edited[2][2] != source[2][2]


def test_stack_undo_redo_and_recipe_round_trip() -> None:
    source = _image(4, (150, 90, 40))
    mask = rect_mask((4, 4), (1, 1, 2, 2))
    stack = EditStack(source)
    stack.add(EditStep("grayscale", {"strength": 1.0}))
    stack.add(EditStep("sepia", {"strength": 0.5}, mask=mask, label="tone"))
    result = stack.apply()
    assert result != source
    assert stack.source == source
    assert stack.history()[1]["label"] == "tone"
    assert stack.undo() is True
    without_sepia = stack.apply()
    assert without_sepia != result
    assert stack.redo() is True
    assert stack.apply() == result
    rebuilt = EditStack.from_recipe(source, stack.to_recipe())
    assert rebuilt.apply() == result
    before, after = stack.before_after()
    assert before == source and after == result


def test_recipe_without_steps_is_refused() -> None:
    blocked = False
    try:
        edit_image({"image": _image(), "recipe": []})
    except ValueError:
        blocked = True
    assert blocked


def test_filter_specialist_reports_the_recipe_and_keeps_the_source() -> None:
    source = _image(6, (180, 120, 60))
    before = deepcopy(source)
    result = edit_image({
        "image": source,
        "recipe": [
            {"name": "grayscale", "parameters": {"strength": 0.5}},
            {"name": "brightness_contrast", "parameters": {"contrast": 0.2}},
        ],
    })
    assert result.output["status"] == "ready"
    assert result.output["source_preserved"] is True
    assert result.output["certified"] is False
    assert len(result.output["steps"]) == 2
    assert source == before
    assert result.output["image"] != source


def test_magic_eraser_previews_or_escalates() -> None:
    plane = [[(20 + 3 * r + 2 * c, 30 + r + c, 80) for c in range(24)] for r in range(24)]
    mask = rect_mask((24, 24), (9, 9, 5, 5))
    preview = erase_region({"image": plane, "mask": mask})
    assert preview.abstained is False
    assert preview.output["status"] == "preview"
    assert preview.output["source_preserved"] is True
    # A planar patch is recoverable, so the fill lands on the true value: what the
    # test must hold is that nothing outside the selection moved.
    assert preview.output["image"][0][0] == plane[0][0]
    assert preview.output["image"][23][23] == plane[23][23]
    filled = preview.output["image"][11][11]
    assert all(abs(filled[i] - plane[11][11][i]) <= 8 for i in range(3))


def test_magic_eraser_refuses_an_empty_selection() -> None:
    plane = _image(8)
    result = erase_region({"image": plane, "mask": [[0.0] * 8 for _ in range(8)]})
    assert result.abstained is True
    assert result.output["reason"] == "empty_selection"


def test_magic_eraser_escalates_on_texture() -> None:
    import random

    rng = random.Random(719)
    noise = [[tuple(rng.randrange(256) for _ in range(3)) for _ in range(24)] for _ in range(24)]
    mask = rect_mask((24, 24), (9, 9, 5, 5))
    result = erase_region({"image": noise, "mask": mask})
    assert result.abstained is True
    assert result.output["reason"] in {"escalate", "router_abstained", "candidate_not_confirmed"}
