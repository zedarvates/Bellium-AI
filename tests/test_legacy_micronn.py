from bellium.micro_nn.features import error_classifier_values, featurize
from bellium.micro_nn.legacy import classify_legacy, list_legacy_models, load_legacy_model


def test_all_legacy_models_load_and_predict() -> None:
    names = list_legacy_models()
    assert len(names) == 11
    for name in names:
        model = load_legacy_model(name)
        assert model["layers"][0] > 0
        # named zeros via schema
        from bellium.micro_nn.features import feature_names
        values = {key: 0.0 for key in feature_names(name)}
        result = classify_legacy(name, values)
        assert result.authority_mode.value == "observe"
        assert result.specialist_id.endswith(":legacy-botte")
        assert "probabilities" in result.output


def test_error_classifier_valueerror_is_runtime() -> None:
    text = "Traceback (most recent call last):\nValueError: invalid literal for int()"
    values = error_classifier_values(text, exit_code=1)
    result = classify_legacy("error_classifier", values)
    assert result.abstained is False
    assert result.output["label"] == "runtime"


def test_featurize_rejects_unknown_fields() -> None:
    try:
        featurize("binary_router", {"complexity": 0.1, "budget_ratio": 0.2, "nope": 1.0})
    except ValueError as exc:
        assert "extra" in str(exc)
    else:
        raise AssertionError("expected mismatch")
