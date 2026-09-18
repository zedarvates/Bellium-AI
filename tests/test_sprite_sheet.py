from copy import deepcopy

from bellium.specialists.sprite_sheet import prepare_sprite_sheet

CELL = 24
BACKGROUND = (255, 255, 255)
FOREGROUND = (30, 40, 60)


def _sheet(offsets, *, cols=None, rows=None, cell=CELL, half=4):
    cols = cols or len(offsets)
    rows = rows or 1
    width = cols * cell
    height = rows * cell
    sheet = [[BACKGROUND for _ in range(width)] for _ in range(height)]
    for index, (dx, dy) in enumerate(offsets):
        column = index % cols
        row = index // cols
        center_x = column * cell + cell // 2
        center_y = row * cell + cell // 2
        for r in range(half * 2 + 1):
            for c in range(half * 2 + 1):
                y = center_y - half + dy + r
                x = center_x - half + dx + c
                if 0 <= y < height and 0 <= x < width:
                    sheet[y][x] = FOREGROUND
    return sheet


def test_grid_is_measured_and_frames_reported() -> None:
    sheet = _sheet([(0, 0), (3, 0), (6, 0), (0, 0)])
    original = deepcopy(sheet)
    result = prepare_sprite_sheet({"family": "ui", "image": sheet})
    assert result.abstained is False
    assert result.output["status"] == "ready"
    assert result.output["grid_source"] == "measured"
    assert result.output["frame_count"] == 4
    assert result.output["cell"] == {"w": CELL, "h": CELL}
    assert sheet == original
    assert result.output["source_preserved"] is True
    phases = [frame["phase"] for frame in result.output["frames"]]
    assert phases[0] == "start"
    assert all(phase is not None for phase in phases[1:])
    assert result.output["loop"]["verdict"] == "loops"


def test_drifting_clip_is_reported() -> None:
    sheet = _sheet([(0, 0), (3, 0), (5, 0), (7, 0)])
    result = prepare_sprite_sheet({"family": "ui", "image": sheet})
    assert result.output["status"] == "ready"
    assert result.output["loop"]["verdict"] == "drifts"
    assert "loop_drifts" in result.output["warnings"]


def test_declared_cell_size_is_verified_and_used() -> None:
    sheet = _sheet([(0, 0), (3, 0), (6, 0), (0, 0)])
    result = prepare_sprite_sheet({"family": "ui", "image": sheet, "cell_size": [CELL, CELL]})
    assert result.output["grid_source"] == "declared"
    assert result.output["frame_count"] == 4


def test_declared_cell_size_that_does_not_divide_is_refused() -> None:
    sheet = _sheet([(0, 0), (3, 0), (6, 0)])
    blocked = False
    try:
        prepare_sprite_sheet({"family": "ui", "image": sheet, "cell_size": [20, CELL]})
    except ValueError:
        blocked = True
    assert blocked


def test_two_row_sheet_reports_rows_and_columns() -> None:
    sheet = _sheet([(0, 0), (3, 0), (0, 3), (3, 3)], cols=2, rows=2)
    result = prepare_sprite_sheet({"family": "ui", "image": sheet})
    assert result.output["frame_count"] == 4
    cells = [(frame["row"], frame["column"]) for frame in result.output["frames"]]
    assert cells == [(0, 0), (0, 1), (1, 0), (1, 1)]


def test_single_frame_sheet_abstains() -> None:
    sheet = _sheet([(0, 0)])
    result = prepare_sprite_sheet({"family": "ui", "image": sheet})
    assert result.abstained is True
    assert result.output["reason"] == "grid_not_confirmed"
    assert result.output["frames"] == []


def test_unknown_family_abstains() -> None:
    sheet = _sheet([(0, 0), (3, 0)])
    result = prepare_sprite_sheet({"family": "portrait", "image": sheet})
    assert result.abstained is True
    assert result.output["reason"] == "unknown_family"


def test_pixels_are_only_returned_on_request() -> None:
    sheet = _sheet([(0, 0), (3, 0)])
    plain = prepare_sprite_sheet({"family": "ui", "image": sheet})
    assert "image" not in plain.output["frames"][0]
    with_pixels = prepare_sprite_sheet({"family": "ui", "image": sheet, "include_pixels": True})
    frame = with_pixels.output["frames"][0]["image"]
    assert len(frame) == CELL and len(frame[0]) == CELL


def test_frame_phases_carry_the_rule_baseline_when_no_model_is_shipped() -> None:
    sheet = _sheet([(0, 0), (3, 0), (6, 0), (0, 0)])
    result = prepare_sprite_sheet({"family": "ui", "image": sheet})
    phases = result.output["frames"][1:]
    assert phases
    for frame in phases:
        assert frame["phase_status"] in {"baseline_only", "suggest", "abstain"}
        assert frame["phase"] is not None
        if frame["phase_status"] == "baseline_only":
            assert frame["phase"] == frame["phase_baseline"]
