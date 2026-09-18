"""Atlas packing: invariants first, then the measured comparison with the baselines."""

from copy import deepcopy
import json

import pytest

from bellium.atlas.packing import (
    MAX_PAGES,
    NO_ROOM,
    OVERSIZED,
    Frame,
    Placement,
    cells,
    check_plan,
    pack,
    parse_frames,
    power_of_two_ceiling,
)
from bellium.specialists.atlas import pack_atlas

MIXED = [
    ("idle_0", 24, 32), ("idle_1", 24, 32), ("run_0", 28, 24), ("run_1", 28, 24),
    ("jump_0", 20, 40), ("jump_1", 20, 40), ("hit_0", 36, 36), ("fall_0", 16, 48),
    ("fall_1", 16, 48), ("fx_0", 48, 12), ("fx_1", 48, 12), ("icon_0", 12, 12),
]
# Measured case: insertion-order shelf leaves 4 frames unplaced on one 96 x 96 page
# where the shipped skyline packer places all 14 (see docs/ATLAS_PACKING_GATE.md).
PINNED = [
    ("f_0", 16, 12), ("f_1", 32, 24), ("f_2", 8, 40), ("f_3", 24, 16), ("f_4", 32, 40),
    ("f_5", 8, 32), ("f_6", 40, 16), ("f_7", 24, 16), ("f_8", 12, 16), ("f_9", 40, 12),
    ("f_10", 24, 24), ("f_11", 40, 24), ("f_12", 16, 32), ("f_13", 12, 24),
]


def _frames(spec):
    return parse_frames([{"name": name, "width": w, "height": h} for name, w, h in spec])


def _overlaps(left, right) -> bool:
    return left[0] < right[2] and right[0] < left[2] and left[1] < right[3] and right[1] < left[3]


@pytest.mark.parametrize("method", ["skyline", "shelf", "next-fit"])
def test_every_method_keeps_the_invariants(method) -> None:
    frames = _frames(MIXED)
    plan = pack(frames, width=128, height=128, method=method)
    assert check_plan(frames, plan) == []
    placed = [item for page in plan.pages for item in page]
    assert {item.name for item in placed} == {frame.name for frame in frames}
    assert plan.complete
    assert plan.unplaced == ()


def test_padding_keeps_a_gutter_from_borders_and_neighbours() -> None:
    frames = _frames(MIXED)
    plan = pack(frames, width=128, height=128, padding=2)
    assert check_plan(frames, plan) == []
    placed = [item for page in plan.pages for item in page]
    assert min(item.x for item in placed) >= 2
    assert min(item.y for item in placed) >= 2
    assert all(item.x + item.width + 2 <= 128 for item in placed)
    assert all(item.y + item.height + 2 <= 128 for item in placed)
    boxes = [cells(item, 2) for item in placed]
    for first in range(len(boxes)):
        for second in range(first + 1, len(boxes)):
            assert not _overlaps(boxes[first], boxes[second])


def test_padding_costs_page_area_and_stays_valid() -> None:
    frames = parse_frames([{"name": f"q_{i}", "width": 8, "height": 8} for i in range(4)])
    tight = pack(frames, width=16, height=16)
    assert tight.complete and tight.occupancy == 1.0
    squeezed = pack(frames, width=16, height=16, padding=1)
    assert squeezed.complete is False
    assert [item.reason for item in squeezed.unplaced] == [NO_ROOM] * 3
    spread = pack(frames, width=16, height=16, padding=1, max_pages=4)
    assert spread.complete and spread.page_count == 4
    assert spread.occupancy == 0.25
    assert check_plan(frames, spread) == []


def test_skyline_does_not_depend_on_the_declared_order() -> None:
    frames = _frames(MIXED)
    plan = pack(frames, width=128, height=128)
    reversed_plan = pack(list(reversed(frames)), width=128, height=128)
    assert plan.to_dict() == reversed_plan.to_dict()


def test_the_naive_baseline_loses_on_the_measured_case() -> None:
    frames = _frames(PINNED)
    skyline = pack(frames, width=96, height=96)
    sorted_shelf = pack(frames, width=96, height=96, method="shelf")
    naive = pack(frames, width=96, height=96, method="next-fit")
    assert skyline.complete and check_plan(frames, skyline) == []
    assert naive.complete is False
    assert {item.name for item in naive.unplaced} == {"f_10", "f_11", "f_12", "f_13"}
    assert round(skyline.occupancy, 4) == 0.7847
    assert round(naive.occupancy, 4) == 0.5312
    assert round(sorted_shelf.occupancy, 4) == round(skyline.occupancy, 4)


def test_a_frame_larger_than_the_page_is_reported_oversized() -> None:
    frames = parse_frames([
        {"name": "huge", "width": 512, "height": 512},
        {"name": "ok", "width": 8, "height": 8},
    ])
    plan = pack(frames, width=64, height=64)
    assert plan.complete is False
    assert [(item.name, item.reason) for item in plan.unplaced] == [("huge", OVERSIZED)]
    assert check_plan(frames, plan) == []


def test_a_full_page_reports_no_room_instead_of_scaling() -> None:
    frames = parse_frames([{"name": f"s_{i}", "width": 8, "height": 8} for i in range(8)])
    one_page = pack(frames, width=16, height=16)
    assert one_page.page_count == 1
    assert [item.reason for item in one_page.unplaced] == [NO_ROOM] * 4
    assert all(item.width == 8 and item.height == 8 for page in one_page.pages for item in page)
    two_pages = pack(frames, width=16, height=16, max_pages=2)
    assert two_pages.complete
    assert two_pages.page_count == 2
    assert two_pages.occupancy == 1.0


def test_occupancy_is_measured_against_the_pages_used() -> None:
    single = parse_frames([{"name": "one", "width": 8, "height": 8}])
    plan = pack(single, width=16, height=16)
    assert plan.used_area == 64
    assert plan.occupancy == 0.25
    full = parse_frames([{"name": f"q_{i}", "width": 8, "height": 8} for i in range(4)])
    assert pack(full, width=16, height=16).occupancy == 1.0


@pytest.mark.parametrize("value,expected", [
    (1, 1), (2, 2), (3, 4), (96, 128), (128, 128), (1000, 1024),
])
def test_power_of_two_ceiling(value, expected) -> None:
    assert power_of_two_ceiling(value) == expected


@pytest.mark.parametrize("value", [0, -4, 2.5, True, None])
def test_power_of_two_ceiling_refuses_invalid_extents(value) -> None:
    with pytest.raises(ValueError):
        power_of_two_ceiling(value)


@pytest.mark.parametrize("frames", [
    [], "frames", None,
    [{"name": "a", "width": 8}],
    [{"name": "a", "height": 8}],
    [{"width": 8, "height": 8}],
    [{"name": "a", "width": 8, "height": 8, "bleed": 1}],
    [{"name": "a", "width": 8, "height": 8}, {"name": "a", "width": 4, "height": 4}],
    [{"name": "  ", "width": 8, "height": 8}],
    [{"name": "a", "width": 0, "height": 8}],
    [{"name": "a", "width": 8, "height": -1}],
    [{"name": "a", "width": 8.0, "height": 8}],
    [{"name": "a", "width": True, "height": 8}],
    [{"name": "a", "width": 8, "height": None}],
    [7],
])
def test_declared_frames_are_validated(frames) -> None:
    with pytest.raises(ValueError):
        parse_frames(frames)


@pytest.mark.parametrize("name,width,height", [("", 8, 8), ("a", 0, 8), ("a", 8, 0), ("a", 8.0, 8)])
def test_frame_contract_is_enforced_directly(name, width, height) -> None:
    with pytest.raises(ValueError):
        Frame(name=name, width=width, height=height)


@pytest.mark.parametrize("kwargs", [
    {"width": 0, "height": 16},
    {"width": 16, "height": -1},
    {"width": 16.0, "height": 16},
    {"width": True, "height": 16},
    {"width": 16, "height": 16, "padding": -1},
    {"width": 16, "height": 16, "padding": 1.5},
    {"width": 16, "height": 16, "padding": True},
    {"width": 16, "height": 16, "max_pages": 0},
    {"width": 16, "height": 16, "max_pages": MAX_PAGES + 1},
    {"width": 16, "height": 16, "method": "best-fit"},
])
def test_packing_arguments_are_validated(kwargs) -> None:
    frames = parse_frames([{"name": "a", "width": 8, "height": 8}])
    with pytest.raises(ValueError):
        pack(frames, **kwargs)


def test_the_specialist_returns_a_serializable_plan_and_keeps_the_input() -> None:
    query = {"frames": [{"name": n, "width": w, "height": h} for n, w, h in PINNED],
             "width": 96, "height": 96, "power_of_two": True}
    before = deepcopy(query)
    result = pack_atlas(query)
    assert result.abstained is False
    assert result.output["status"] == "ready"
    assert result.output["page_width"] == 128 and result.output["page_height"] == 128
    assert result.output["page_count"] == 1
    assert result.output["complete"] is True
    assert result.output["pixels_written"] is False
    assert result.output["certified"] is False
    assert result.output["frame_count"] == len(PINNED)
    assert result.output["unplaced_count"] == 0
    assert result.confidence == 0.9
    assert json.dumps(result.output)
    assert query == before


def test_the_specialist_abstains_when_frames_do_not_fit() -> None:
    result = pack_atlas({"frames": [{"name": "huge", "width": 640, "height": 480}],
                         "width": 256, "height": 256})
    assert result.abstained is True
    assert result.output["status"] == "abstain"
    assert result.output["reason"] == "frames_do_not_fit"
    assert result.output["unplaced"] == [{"name": "huge", "reason": OVERSIZED}]
    assert result.output["pages"] == []
    assert result.confidence == 0.0
    assert result.output["certified"] is False


@pytest.mark.parametrize("query", [
    {"width": 64, "height": 64},
    {"frames": [{"name": "a", "width": 8, "height": 8}], "height": 64},
    {"frames": [{"name": "a", "width": 8, "height": 8}], "width": 64},
    {"frames": [{"name": "a", "width": 8, "height": 8}], "width": 64, "height": 64, "gutter": 2},
    {"frames": [{"name": "a", "width": 8, "height": 8}], "width": 64, "height": 64,
     "power_of_two": "yes"},
])
def test_the_specialist_refuses_incomplete_queries(query) -> None:
    with pytest.raises(ValueError):
        pack_atlas(query)


def test_the_specialist_refuses_a_non_object_query() -> None:
    with pytest.raises(ValueError):
        pack_atlas(None)


def test_placements_are_frozen_geometry() -> None:
    placement = Placement(name="a", page=0, x=0, y=0, width=8, height=8)
    assert cells(placement, 1) == (-1, -1, 9, 9)
    with pytest.raises(Exception):
        placement.x = 4
