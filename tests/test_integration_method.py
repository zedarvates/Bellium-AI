"""The integrator recommendation: measured labels, a published rule beside it."""

import copy

import pytest

from bellium.contracts.schema import AuthorityMode
from bellium.knn.integration_method import (
    FEATURE_NAMES,
    METHODS,
    MIN_SIMILARITY,
    SCHEMA,
    deterministic_method,
    integration_features,
    load_memory,
    measured_label,
    recommend_method,
    similarity,
)
from bellium.material.controlled import perturbed_normals
from bellium.material.integration import analytic_height, height_error, integrate_height
from bellium.material.photometric import synthetic_geometry
from bellium.specialists.integration_method import recommend_integration

SIZE = 32
SCALE = 1.0 / (SIZE * 0.45)


def _field(geometry, sigma=0.0, seed=1):
    normals, mask = synthetic_geometry(geometry, size=SIZE)
    return perturbed_normals(normals, sigma, seed=seed), mask


def test_features_are_bounded_and_separate_the_surfaces() -> None:
    features = {
        name: integration_features(*_field(name))
        for name in ("tilted-plane", "sphere", "cone", "waves")
    }
    for values in features.values():
        assert set(values) == set(FEATURE_NAMES)
        assert all(0.0 <= value <= 1.0 for value in values.values())
    plane = features["tilted-plane"]
    assert plane["direction_focus"] == 1.0
    assert plane["divergence_ratio"] == 0.0
    assert plane["flat_field"] == 0.0
    # a dome points away from its centre and spreads over the circle
    assert features["sphere"]["radial_alignment"] > 0.9
    assert features["sphere"]["direction_focus"] < 0.05
    # an oscillation diverges where a ramp does not
    assert features["waves"]["divergence_ratio"] > plane["divergence_ratio"] + 0.5
    # a cone keeps a constant gradient magnitude, which is what uniformity sees
    assert features["cone"]["slope_uniformity"] > features["sphere"]["slope_uniformity"]


def test_a_field_without_slope_is_reported_as_flat() -> None:
    normals = [[(0.0, 0.0, 1.0)] * 4 for _ in range(4)]
    features = integration_features(normals)
    assert features["flat_field"] == 1.0
    assert deterministic_method(features) == "cumulative-vertical"


def test_features_require_a_minimum_field() -> None:
    normals = [[(0.0, 0.0, 1.0), (1.0, 0.0, 0.0)], [(1.0, 0.0, 0.0), (1.0, 0.0, 0.0)]]
    assert integration_features(normals) is None


def test_the_published_rule_names_the_vertical_alignment_on_a_ramp() -> None:
    ramp = integration_features(*_field("tilted-plane"))
    dome = integration_features(*_field("sphere"))
    assert deterministic_method(ramp) == "cumulative-vertical"
    assert deterministic_method(dome) == "cumulative-average"


def test_the_shipped_memory_is_complete_and_bounded() -> None:
    memory = load_memory()
    assert memory["schema"] == SCHEMA
    assert len(memory["items"]) == 16
    for item in memory["items"]:
        assert item["method"] in METHODS
        assert set(item["errors"]) == set(METHODS)
        assert min(item["errors"].values()) >= 0.0
        assert all(0.0 <= item["features"][name] <= 1.0 for name in FEATURE_NAMES)
        # the label is the measured winner, with ties broken by declared cost order
        assert item["method"] == measured_label(item["errors"])


def test_the_measured_label_prefers_the_cheaper_method_on_a_tie() -> None:
    tied = {
        "least-squares": 0.0,
        "cumulative": 0.5,
        "cumulative-average": 0.0,
        "cumulative-vertical": 0.0,
    }
    assert measured_label(tied) == "cumulative-vertical"
    clear = {**tied, "cumulative-vertical": 0.25}
    assert measured_label(clear) == "cumulative-average"
    with pytest.raises(ValueError, match="must cover every method"):
        measured_label({"least-squares": 0.0})
    with pytest.raises(ValueError, match="finite and not negative"):
        measured_label(dict.fromkeys(METHODS, -1.0))
    with pytest.raises(ValueError, match="must be numeric"):
        measured_label({**dict.fromkeys(METHODS, 0.0), "cumulative": True})


@pytest.mark.parametrize(
    "broken",
    [{"schema": "wrong", "items": [{"id": "x", "method": "cumulative-average", "features": {}}]},
     {"schema": SCHEMA, "items": []},
     {"schema": SCHEMA, "items": [{"id": "x", "method": "gradient-descent", "features": {}}]}],
)
def test_a_memory_that_does_not_match_the_contract_is_refused(tmp_path, broken) -> None:
    import json

    path = tmp_path / "memory.json"
    path.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(ValueError):
        load_memory(path)


def test_a_memory_with_an_out_of_range_feature_is_refused(tmp_path) -> None:
    import json

    memory = load_memory()
    broken = copy.deepcopy(memory)
    broken["items"][0]["features"][FEATURE_NAMES[0]] = 1.5
    path = tmp_path / "memory.json"
    path.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(ValueError, match="between 0 and 1"):
        load_memory(path)


def test_the_recommendation_matches_the_measured_winner() -> None:
    for geometry, expected in (
        ("sphere", "cumulative-average"),
        ("cone", "cumulative-average"),
        ("waves", "cumulative-average"),
        ("tilted-plane", "cumulative-vertical"),
    ):
        answer = recommend_method(*_field(geometry))
        assert answer["status"] == "ready"
        assert answer["method"] == expected
        assert len(answer["neighbours"]) == 3
        assert answer["agreement"] > 0.0
        scores = [n["similarity"] for n in answer["neighbours"]]
        assert scores == sorted(scores, reverse=True)
        assert all(set(n["errors"]) == set(METHODS) for n in answer["neighbours"])


def test_the_learned_tier_overrides_the_rule_on_a_perturbed_ramp() -> None:
    """The rule cannot see noise: 0.0057 for its answer against 0.0038 for the k-NN."""
    field, mask = _field("tilted-plane", sigma=0.02, seed=2)
    answer = recommend_method(field, mask)
    assert answer["method"] == "least-squares"
    assert answer["baseline_method"] == "cumulative-vertical"
    assert answer["agrees_with_baseline"] is False
    result = recommend_integration({"normals": field, "mask": mask})
    assert "learned_tier_overrides_the_published_rule" in result.output["warnings"]
    truth, truth_mask = analytic_height("tilted-plane", size=SIZE)
    errors = {}
    for method in ("least-squares", "cumulative-vertical"):
        out = integrate_height(field, mask, method=method, iterations=400, pixel_scale=SCALE)
        errors[method] = height_error(out["height"], truth, truth_mask)["relative_rmse"]
    assert errors["least-squares"] < errors["cumulative-vertical"]
    assert answer["method"] == min(errors, key=lambda name: errors[name])


def test_an_unfamiliar_field_falls_back_to_the_published_rule() -> None:
    extremes = [
        {"id": f"far_{index}", "method": "cumulative-average", "features": dict.fromkeys(FEATURE_NAMES, 0.0)}
        for index in range(4)
    ]
    memory = {"schema": SCHEMA, "items": extremes}
    answer = recommend_method(*_field("sphere"), memory=memory)
    assert answer["status"] == "baseline"
    assert answer["reason"] == "not_enough_neighbours"
    assert answer["method"] == answer["baseline_method"]
    result = recommend_integration({"normals": _field("sphere")[0], "memory": memory})
    assert result.abstained is False
    assert result.output["recommended_by"] == "baseline"
    assert result.output["warnings"] == ["not_enough_neighbours"]
    assert result.confidence == 0.5
    assert similarity(answer["features"], extremes[0]["features"]) < MIN_SIMILARITY


def test_the_specialist_abstains_on_a_field_it_cannot_characterise() -> None:
    normals = [[(1.0, 0.0, 0.0), None], [(0.0, 0.0, 0.0), (0.0, 1.0, 0.0)]]
    result = recommend_integration({"normals": normals})
    assert result.abstained is True
    assert result.confidence == 0.0
    assert result.output["reason"] == "field_too_small"
    assert result.output["method"] is None
    assert result.output["features"] is None
    assert result.specialist_id == "bellium/knn/integration-method:v0"


def test_the_specialist_recommends_without_integrating() -> None:
    field, mask = _field("sphere")
    result = recommend_integration({"normals": field, "mask": mask, "min_cosine": 0.1})
    assert result.authority_mode is AuthorityMode.CONSULTATIVE
    output = result.output
    assert output["status"] == "ready"
    assert output["integrated"] is False
    assert output["certified"] is False
    assert output["pixels_changed"] == 0
    assert "height" not in output
    assert output["baseline_id"] == "bellium/deterministic/integration-rule:v0"
    assert output["exemplars"] == 16
    assert 0.0 < result.confidence <= 1.0


def test_the_specialist_refuses_undocumented_keys_and_missing_normals() -> None:
    with pytest.raises(ValueError, match="unknown query keys"):
        recommend_integration({"normals": [[(0.0, 0.0, 1.0)]], "certify": True})
    with pytest.raises(ValueError, match="normals must be provided"):
        recommend_integration({})
    with pytest.raises(ValueError, match="query must be an object"):
        recommend_integration(None)
    with pytest.raises(ValueError, match="memory must be an object"):
        recommend_integration({"normals": [[(0.0, 0.0, 1.0)]], "memory": "shipped"})
    with pytest.raises(ValueError, match="neighbours must be"):
        recommend_method(*_field("sphere"), neighbours=0)
