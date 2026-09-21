import random

import pytest

from bellium.editing import (
    EditStack,
    EditStep,
    apply_filter,
    apply_filter_fast,
    backend_for,
    numpy_available,
)
from bellium.editing.fast import apply_with_tables, channel_tables
from bellium.editing.preview import downsample, downsample_mask, preview_stack
from bellium.editing.preview import PreviewSession

OPS = (
    ("grayscale", {"strength": 0.7}),
    ("grayscale", {"weights": "average"}),
    ("sepia", {"strength": 0.9}),
    ("sepia", {}),
    ("invert", {}),
    ("invert", {"strength": 0.25}),
    ("brightness_contrast", {"brightness": 0.1, "contrast": 0.3}),
    ("saturation", {"factor": 1.5}),
    ("saturation", {"factor": 0.2}),
    ("tint", {"factors": (1.2, 0.9, 0.7)}),
    ("threshold", {"level": 0.4}),
)


def _image(size=12, seed=3):
    rng = random.Random(seed)
    return [[tuple(rng.randrange(256) for _ in range(3)) for _ in range(size)]
            for _ in range(size)]


@pytest.mark.parametrize("name,parameters", OPS)
def test_fast_paths_match_the_plain_path_exactly(name, parameters) -> None:
    image = _image()
    assert apply_filter_fast(image, name, **parameters) == apply_filter(
        image, name, **parameters
    )


@pytest.mark.parametrize("name,parameters", [op for op in OPS if op[0] in
                                             ("invert", "brightness_contrast", "tint")])
def test_lookup_tables_match_the_plain_path(name, parameters) -> None:
    image = _image()
    tables = channel_tables(name, **parameters)
    assert len(tables) == 3 and all(len(table) == 256 for table in tables)
    assert apply_with_tables(image, tables) == apply_filter(image, name, **parameters)


def test_channel_wise_rules_are_declared() -> None:
    for name in ("invert", "brightness_contrast", "tint"):
        assert backend_for(name) in {"numpy", "lut"}
    assert backend_for("sepia") in {"numpy", "pure"}
    blocked = False
    try:
        channel_tables("sepia")
    except ValueError:
        blocked = True
    assert blocked


def test_unknown_filter_is_refused_by_the_fast_path() -> None:
    blocked = False
    try:
        apply_filter_fast(_image(), "oil-paint")
    except ValueError:
        blocked = True
    assert blocked


def test_a_saturated_colour_blends_like_the_plain_path() -> None:
    # A sepia red channel can exceed 255 before clipping; the order of clipping
    # and blending is what the two paths had to agree on.
    image = [[(98, 240, 243), (215, 197, 179)]]
    assert apply_filter_fast(image, "sepia", strength=0.9) == apply_filter(
        image, "sepia", strength=0.9
    )


def test_downsampling_is_an_area_average() -> None:
    flat = [[(120, 130, 140) for _ in range(32)] for _ in range(32)]
    reduced = downsample(flat, max_side=16)
    assert reduced["size"] == (16, 16)
    assert reduced["scale"] == pytest.approx(0.5)
    assert reduced["resampled"] is True
    assert reduced["image"][0][0] == (120, 130, 140)
    untouched = downsample(flat, max_side=64)
    assert untouched["resampled"] is False
    assert untouched["image"] == flat


def test_preview_is_reported_as_a_preview_and_differs_from_the_export() -> None:
    image = _image(48, seed=9)
    stack = EditStack(image)
    stack.add(EditStep("sepia", {"strength": 0.5}))
    preview = preview_stack(stack, max_side=16)
    assert preview["preview_only"] is True
    assert preview["size"] == (16, 16)
    assert preview["export_size"] == (48, 48)
    assert preview["image"] != stack.apply()
    assert len(stack.apply()) == 48


def test_preview_follows_the_recipe_and_the_masks() -> None:
    flat = [[(200, 100, 50) for _ in range(32)] for _ in range(32)]
    mask = [[0.0] * 32 for _ in range(32)]
    for r in range(16):
        for c in range(32):
            mask[r][c] = 1.0
    stack = EditStack(flat)
    stack.add(EditStep("invert", {}, mask=mask))
    preview = preview_stack(stack, max_side=16)
    assert preview["image"][0][0] != flat[0][0]
    assert preview["image"][15][0] == flat[0][0]
    empty = EditStack(flat)
    assert preview_stack(empty, max_side=16)["image"] == downsample(flat, max_side=16)["image"]


def test_mask_downsampling_keeps_the_range() -> None:
    mask = [[1.0] * 32 for _ in range(32)]
    for r in range(16):
        for c in range(32):
            mask[r][c] = 0.0
    reduced = downsample_mask(mask, size=(16, 16))
    assert len(reduced) == 16 and len(reduced[0]) == 16
    assert all(0.0 <= value <= 1.0 for row in reduced for value in row)
    assert reduced[0][0] == 0.0 and reduced[15][15] == 1.0


def test_preview_inputs_are_validated() -> None:
    stack = EditStack(_image(8))
    for call in (
        lambda: preview_stack(stack, max_side=2),
        lambda: preview_stack(stack, max_side=8, backend="gpu"),
        lambda: downsample(_image(8), max_side=1),
    ):
        blocked = False
        try:
            call()
        except ValueError:
            blocked = True
        assert blocked


def test_numpy_is_optional_and_reported() -> None:
    assert isinstance(numpy_available(), bool)
    # Whatever the environment, the plain path is always the fallback contract.
    image = _image(6)
    assert apply_filter(image, "invert") == apply_filter_fast(image, "invert")


def test_preview_session_matches_the_plain_preview() -> None:
    image = _image(32, seed=21)
    steps = (
        EditStep("sepia", {"strength": 0.6}),
        EditStep("brightness_contrast", {"contrast": 0.2}),
    )
    stack = EditStack(image)
    for step in steps:
        stack.add(step)
    plain = preview_stack(stack, max_side=16)
    session = PreviewSession(image, backend="fast" if numpy_available() else "plain")
    session_preview = session.preview(steps, max_side=16)
    assert session_preview["size"] == plain["size"]
    assert session_preview["preview_only"] is True
    assert session_preview["export_size"] == (32, 32)
    assert session_preview["image"] == plain["image"]


def test_preview_session_respects_masks_and_reuse() -> None:
    flat = [[(180, 90, 40) for _ in range(16)] for _ in range(16)]
    mask = [[0.0] * 16 for _ in range(16)]
    for r in range(8):
        for c in range(16):
            mask[r][c] = 1.0
    step = EditStep("invert", {}, mask=mask)
    session = PreviewSession(flat, backend="fast" if numpy_available() else "plain")
    first = session.preview((step,), max_side=16)
    second = session.preview((step,), max_side=16)
    assert first["image"] == second["image"]
    assert first["image"][0][0] != flat[0][0]
    assert first["image"][15][0] == flat[15][0]
