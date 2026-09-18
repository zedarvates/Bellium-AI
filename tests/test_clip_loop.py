from bellium.knn.clip_loop import classify_loop, load_clips

CELL = 24


def _mask(half=4, dx=0):
    center = CELL // 2
    return [
        [
            1 if abs(r - center) <= half and abs(c - center - dx) <= half else 0
            for c in range(CELL)
        ]
        for r in range(CELL)
    ]


def _image(mask, tint=0):
    return [
        [
            (min(255, 210 + tint), 60, 60) if value else (255, 255, 255)
            for value in row
        ]
        for row in mask
    ]


def _pair(dx_last, tint=0):
    first = _mask()
    last = _mask(dx=dx_last)
    return {
        "first": {"image": _image(first), "mask": first},
        "last": {"image": _image(last, tint=tint), "mask": last},
    }


def test_clean_loop_is_confirmed() -> None:
    result = classify_loop({"family": "ui", "frames": _pair(0)})
    assert result.abstained is False
    assert result.output["verdict"] == "loops"
    assert result.output["baseline"] == "loops"
    assert result.output["pixels_changed"] == 0


def test_colour_drift_still_loops() -> None:
    result = classify_loop({"family": "ui", "frames": _pair(0, tint=10)})
    assert result.abstained is False
    assert result.output["verdict"] == "loops"


def test_large_drift_is_reported() -> None:
    result = classify_loop({"family": "ui", "frames": _pair(8)})
    assert result.abstained is False
    assert result.output["verdict"] == "drifts"
    assert result.output["baseline"] == "drifts"


def test_small_drift_has_no_comparable_precedent() -> None:
    result = classify_loop({"family": "ui", "frames": _pair(2)})
    assert result.abstained is True
    assert result.output["reason"] == "between_known_spaces"
    assert result.output["baseline"] == "drifts"


def test_unknown_family_abstains() -> None:
    result = classify_loop({"family": "portrait", "frames": _pair(8)})
    assert result.abstained is True
    assert result.output["reason"] == "unknown_family"


def test_out_of_range_feature_is_refused() -> None:
    blocked = False
    try:
        classify_loop({"family": "ui", "features": {
            "iou_mismatch": 1.4, "added_ratio": 0.1, "removed_ratio": 0.1,
            "centroid_shift": 0.1, "area_ratio": 0.1, "color_delta": 0.1, "bbox_shift": 0.1,
        }})
    except ValueError:
        blocked = True
    assert blocked


def test_raw_pixels_are_not_stored(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(
        '{"schema":"bellium.clip-loop-memory/v1","items":[{'
        '"id":"x","family":"ui","verdict":"loops","source":"fixture","raw_image_stored":true,'
        '"features":{"iou_mismatch":0,"added_ratio":0,"removed_ratio":0,"centroid_shift":0,'
        '"area_ratio":0,"color_delta":0,"bbox_shift":0}}]}',
        encoding="utf-8",
    )
    blocked = False
    try:
        load_clips(path)
    except ValueError:
        blocked = True
    assert blocked
