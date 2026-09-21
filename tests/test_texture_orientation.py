import math

from bellium.knn._texture import (
    axis_of,
    direction_scan,
    local_period_variation,
    shift_period,
)
from bellium.knn.texture_orientation import (
    classify_orientation,
    deterministic_grain_verdict,
    load_orientations,
)

SIZE = 64


def _gray(value):
    level = max(0, min(255, int(value)))
    return level, level, level


def _stripes(period=12, angle_deg=0.0, amplitude=90.0):
    radians = math.radians(angle_deg)
    return [
        [
            _gray(128 + amplitude * math.sin(
                2 * math.pi * (c * math.cos(radians) + r * math.sin(radians)) / period
            ))
            for c in range(SIZE)
        ]
        for r in range(SIZE)
    ]


def _checker(cell=8):
    return [
        [_gray(200 if ((r // cell) + (c // cell)) % 2 else 60) for c in range(SIZE)]
        for r in range(SIZE)
    ]


def _wave2d(period=16):
    return [
        [
            _gray(128 + 70 * math.sin(2 * math.pi * c / period)
                  + 70 * math.cos(2 * math.pi * r / period))
            for c in range(SIZE)
        ]
        for r in range(SIZE)
    ]


def _noise(seed=5):
    import random

    rng = random.Random(seed)
    return [[_gray(rng.randrange(256)) for _ in range(SIZE)] for _ in range(SIZE)]


def _warped(image, strength=0.35):
    out = []
    for r in range(SIZE):
        scale = 1.0 + strength * (r / SIZE)
        out.append([
            image[r][min(max(int(round(c / scale)), 0), SIZE - 1)] for c in range(SIZE)
        ])
    return out


def test_shift_period_finds_the_true_period_not_a_multiple() -> None:
    assert shift_period(_stripes(period=8), 0.0)[0] == 8
    assert shift_period(_stripes(period=12), 0.0)[0] == 12


def test_a_direction_without_variation_declares_no_period() -> None:
    assert shift_period(_stripes(period=12), 90.0)[0] is None


def test_direction_scan_separates_axes_tiles_and_diagonals() -> None:
    scan = direction_scan(_stripes(period=12), None)
    assert scan["axis_x"]["period"] == 12 and scan["axis_y"] is None
    tile = direction_scan(_checker(cell=8), None)
    assert tile["axis_x"]["period"] == 16 and tile["axis_y"]["period"] == 16
    diagonal = direction_scan(_stripes(period=12, angle_deg=45.0), None)
    assert diagonal["axis_x"] is None and diagonal["diagonal"]["period"] == 12
    assert direction_scan(_noise(), None)["best"] is None


def test_axis_of_classifies_directions() -> None:
    assert axis_of(0.0) == "x"
    assert axis_of(175.0) == "x"
    assert axis_of(90.0) == "y"
    assert axis_of(83.0) == "y"
    assert axis_of(45.0) is None


def test_grain_verdict_prefers_axes_over_a_shorter_diagonal() -> None:
    assert deterministic_grain_verdict(direction_scan(_checker(cell=8), None)) == "repeat-both"
    assert deterministic_grain_verdict(
        direction_scan(_stripes(period=12, angle_deg=40.0), None)
    ) == "repeat-diagonal"


def test_local_variation_exposes_a_perspective_warp() -> None:
    straight = local_period_variation(_stripes(period=12), 0.0)
    assert straight["varies"] is False
    warped = local_period_variation(_warped(_stripes(period=12)), 0.0)
    assert warped["varies"] is True
    assert warped["ratio"] > 1.15


def test_orientation_is_reported_for_known_grains() -> None:
    assert classify_orientation({"family": "texture", "image": _stripes(period=12)}).output["grain"] == "repeat-x"
    assert classify_orientation({"family": "texture", "image": _checker(cell=8)}).output["grain"] == "repeat-both"
    assert classify_orientation({"family": "texture", "image": _wave2d()}).output["grain"] == "repeat-both"
    diagonal = classify_orientation({"family": "texture", "image": _stripes(period=12, angle_deg=45.0)})
    assert diagonal.output["grain"] == "repeat-diagonal"
    assert abs(diagonal.output["angle_deg"] - 45.0) <= 7.0
    assert classify_orientation({"family": "texture", "image": _noise()}).output["grain"] == "no-repeat"


def test_perspective_abstains_instead_of_reporting_one_period() -> None:
    result = classify_orientation({"family": "texture", "image": _warped(_stripes(period=12))})
    assert result.abstained is True
    assert result.output["reason"] == "repeat_varies_across_image"
    assert result.output["grain"] is None


def test_unknown_family_abstains() -> None:
    result = classify_orientation({"family": "portrait", "image": _stripes()})
    assert result.abstained is True
    assert result.output["reason"] == "unknown_family"


def test_memory_refuses_stored_pixels(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(
        '{"schema":"bellium.texture-orientation-memory/v1","items":[{'
        '"id":"x","family":"texture","grain":"repeat-x","source":"fixture",'
        '"raw_image_stored":true,"features":{}}]}',
        encoding="utf-8",
    )
    blocked = False
    try:
        load_orientations(path)
    except ValueError:
        blocked = True
    assert blocked
