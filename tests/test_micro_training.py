import copy
import math

import pytest

from bellium.micro_nn.mlp import predict_mlp, validate_mlp
from bellium.micro_nn.train import train_classifier, train_regressor


CLASSIFICATION = [([0.0], 0), ([0.2], 0), ([0.8], 1), ([1.0], 1)]
REGRESSION = [([x], 2.0 * x + 1.0) for x in (0.0, 0.2, 0.8, 1.0)]


def test_classifier_preserves_samples_and_is_reproducible():
    samples = copy.deepcopy(CLASSIFICATION)
    original = copy.deepcopy(samples)
    first = train_classifier(samples, layer_sizes=[1, 3, 2], epochs=20)
    assert samples == original
    second = train_classifier(samples, layer_sizes=[1, 3, 2], epochs=20)
    assert first == second


@pytest.mark.parametrize("trainer,samples,output", [
    (train_classifier, CLASSIFICATION, 2), (train_regressor, REGRESSION, 1),
])
def test_early_stopping_restores_best_epoch(trainer, samples, output):
    original = copy.deepcopy(samples)
    references = [trainer(samples, layer_sizes=[1, 3, output], epochs=epoch)
                  for epoch in (1, 2, 3)]
    def loss(candidate):
        if output == 1:
            return sum((predict_mlp(candidate, vec)[0] - target) ** 2
                       for vec, target in samples)
        return sum(-math.log(predict_mlp(candidate, vec)[target]) for vec, target in samples)
    best_index = min(range(3), key=lambda index: loss(references[index]))
    reference = references[best_index]
    model = trainer(
        samples, layer_sizes=[1, 3, output], epochs=20,
        validation_samples=samples, patience=2, min_delta=1e6,
    )
    assert samples == original
    assert model["weights"] == reference["weights"]
    assert model["biases"] == reference["biases"]
    assert model["training"]["epochs_run"] == 3
    assert model["training"]["best_epoch"] == best_index + 1
    assert model["training"]["stopped_early"] is True
    validate_mlp(model)
    replay = trainer(
        samples, layer_sizes=[1, 3, output], epochs=20,
        validation_samples=samples, patience=2, min_delta=1e6,
    )
    assert replay == model


@pytest.mark.parametrize("trainer,samples,output", [
    (train_classifier, CLASSIFICATION, 2), (train_regressor, REGRESSION, 1),
])
def test_validation_only_restores_best_and_runs_all_epochs(trainer, samples, output):
    model = trainer(samples, layer_sizes=[1, 3, output], epochs=5, validation_samples=samples)
    assert model["training"]["epochs_run"] == 5
    assert model["training"]["stopped_early"] is False
    assert 1 <= model["training"]["best_epoch"] <= 5


@pytest.mark.parametrize("trainer,samples,output", [
    (train_classifier, CLASSIFICATION, 2), (train_regressor, REGRESSION, 1),
])
def test_validation_data_is_checked_before_training(trainer, samples, output):
    with pytest.raises(ValueError, match="features"):
        trainer(samples, layer_sizes=[1, 3, output], validation_samples=[([0.0, 1.0], 0)])


@pytest.mark.parametrize("trainer,samples,output", [
    (train_classifier, CLASSIFICATION, 2), (train_regressor, REGRESSION, 1),
])
@pytest.mark.parametrize("kwargs", [
    {"epochs": 0}, {"epochs": True}, {"lr": 0.0}, {"lr": float("nan")},
    {"patience": 2}, {"validation_samples": []},
    {"validation_samples": [([0.5], 0)], "patience": True},
    {"min_delta": -1.0},
])
def test_invalid_training_controls_are_rejected(trainer, samples, output, kwargs):
    with pytest.raises(ValueError):
        trainer(samples, layer_sizes=[1, 3, output], **kwargs)


@pytest.mark.parametrize("samples,layers", [
    ([([0.0, 1.0], 0)], [1, 2]), ([([float("nan")], 0)], [1, 2]),
    ([([True], 0)], [1, 2]), ([([0.0], -1)], [1, 2]),
    ([([0.0], 2)], [1, 2]), ([([0.0], True)], [1, 2]),
    (CLASSIFICATION, [0, 2]), (CLASSIFICATION, [1]),
])
def test_invalid_classifier_data_is_rejected(samples, layers):
    with pytest.raises(ValueError):
        train_classifier(samples, layer_sizes=layers, epochs=1)


@pytest.mark.parametrize("samples,layers", [
    ([([0.0], float("inf"))], [1, 1]), ([([0.0], True)], [1, 1]),
    (REGRESSION, [1, 2]), ([([0.0, 1.0], 1.0)], [1, 1]),
])
def test_invalid_regression_data_is_rejected(samples, layers):
    with pytest.raises(ValueError):
        train_regressor(samples, layer_sizes=layers, epochs=1)


def test_regressor_learns_scalar_relation():
    model = train_regressor(REGRESSION, layer_sizes=[1, 1], epochs=100, lr=0.03)
    assert abs(predict_mlp(model, [0.5])[0] - 2.0) < 0.05


@pytest.mark.parametrize("trainer,samples,output", [
    (train_classifier, [([1e308], 0)], 2),
    (train_regressor, [([1e308], 1e308)], 1),
])
def test_training_overflow_is_rejected(trainer, samples, output):
    with pytest.raises(ValueError, match="finite|overflow"):
        trainer(samples, layer_sizes=[1, 3, output], epochs=2)
