from bellium.knn.tileability import classify_tileability


def _gray(value):
    level = max(0, min(255, int(value)))
    return (level, level, level)


def _wrapped_step(size=64, step=120):
    return [
        [_gray(90 + (step / 2 if c < size // 2 else -step / 2)) for c in range(size)]
        for _ in range(size)
    ]


def _tiled(size=64, tile=16, seed=3):
    import random

    rng = random.Random(seed)
    block = [[_gray(rng.randrange(256)) for _ in range(tile)] for _ in range(tile)]
    return [[block[r % tile][c % tile] for c in range(size)] for r in range(size)]


def test_wrap_discontinuity_is_reported_on_its_axis() -> None:
    result = classify_tileability({"family": "texture", "image": _wrapped_step()})
    assert result.abstained is False
    assert result.output["verdict"] == "seam"
    assert result.output["seam_axes"] == ["x"]
    assert result.output["pixels_changed"] == 0


def test_tiled_repetition_is_continuous() -> None:
    result = classify_tileability({"family": "tilemap", "image": _tiled()})
    assert result.abstained is False
    assert result.output["verdict"] == "tileable"
    assert result.output["seam_axes"] == []


def test_unknown_family_abstains() -> None:
    result = classify_tileability({"family": "portrait", "image": _tiled()})
    assert result.abstained is True
    assert result.output["reason"] == "unknown_family"


def test_conflicting_neighbors_abstain() -> None:
    vector = {"gap": 0.9, "seam": 0.9, "interior": 0.1, "edge_step": 0.05,
              "seam_max": 0.9, "hot_rows": 0.9, "axis_asym": 1.0}
    conflicting = []
    for axis in ("x", "y"):
        for index in range(6):
            conflicting.append({
                "id": f"conflict_{axis}_{index}",
                "family": "texture",
                "axis": axis,
                "verdict": "seam" if index % 2 else "tileable",
                "source": "fixture:test",
                "features": dict(vector),
            })
    result = classify_tileability(
        {"family": "texture", "image": _wrapped_step()}, memory=conflicting
    )
    assert result.abstained is True
    assert result.output["axes"]["x"]["reason"] == "mixed_neighbor_verdicts"


def test_unknown_feature_is_not_coerced() -> None:
    blocked = False
    try:
        classify_tileability({"family": "texture", "features": {"x": {"gap": 0.5}}})
    except ValueError:
        blocked = True
    assert blocked
