import math

from bellium.knn.texture_repeat import classify_repeat, load_repeats


def _sine(period=8, size=64, amplitude=90.0):
    return [
        [
            (
                max(0, min(255, int(128 + amplitude * math.sin(2 * math.pi * c / period)))),
            ) * 3
            for c in range(size)
        ]
        for _ in range(size)
    ]


def _noise(size=64, seed=5):
    import random

    rng = random.Random(seed)
    return [[(rng.randrange(256),) * 3 for _ in range(size)] for _ in range(size)]


def test_measured_period_is_reported() -> None:
    result = classify_repeat({"family": "texture", "image": _sine(period=8)})
    assert result.abstained is False
    assert result.output["verdict"] == "periodic"
    assert result.output["period_x"] == 8
    assert result.output["pixels_changed"] == 0


def test_noise_is_not_declared_periodic() -> None:
    result = classify_repeat({"family": "texture", "image": _noise()})
    assert result.output.get("verdict") != "periodic"
    if not result.abstained:
        assert result.output["verdict"] == "nonperiodic"
        assert result.output["period_x"] is None


def test_repeat_counts_need_a_declared_extent() -> None:
    without = classify_repeat({"family": "texture", "image": _sine(period=8)})
    assert without.output["repeat_counts"] is None
    declared = classify_repeat(
        {"family": "texture", "image": _sine(period=8), "target_extent_px": [256, 128]}
    )
    assert declared.output["repeat_counts"] == {
        "x": 32,
        "target_extent_px": [256.0, 128.0],
    }


def test_unknown_family_abstains() -> None:
    result = classify_repeat({"family": "portrait", "image": _sine()})
    assert result.abstained is True
    assert result.output["reason"] == "unknown_family"


def test_periodic_item_must_declare_a_period(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(
        '{"schema":"bellium.texture-repeat-memory/v1","items":[{'
        '"id":"x","family":"texture","verdict":"periodic","source":"fixture",'
        '"features":{"strength_x":1,"strength_y":1,"axis_balance":1,'
        '"gradient_energy":0.5,"profile_variance":0.5,"detail_ratio":0.01}}]}',
        encoding="utf-8",
    )
    blocked = False
    try:
        load_repeats(path)
    except ValueError:
        blocked = True
    assert blocked


def test_unknown_feature_is_not_coerced() -> None:
    blocked = False
    try:
        classify_repeat({"family": "texture", "features": {"strength_x": 0.5}})
    except ValueError:
        blocked = True
    assert blocked
