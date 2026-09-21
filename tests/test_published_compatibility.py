"""Keep the published entry points covered by the normal test command."""
import importlib

import pytest


@pytest.mark.parametrize("module", [
    "legacy.micro_nn.test_features", "legacy.micro_nn.test_botte_nn",
    "legacy.micro_nn.test_calibration", "legacy.knn_asset_quality.test_asset_quality",
])
def test_legacy_checks(module, tmp_path, monkeypatch):
    if module.endswith("test_calibration"):
        from legacy.micro_nn import calibration
        monkeypatch.setattr(calibration, "_MODELS_DIR", tmp_path)
        monkeypatch.setattr(calibration, "_cache", {})
    assert importlib.import_module(module).main() == 0


@pytest.mark.parametrize("module", [
    "bellium.cutout.test_cutout", "bellium.inpaint.test_inpaint",
    "bellium.language.test_phoneme", "bellium.routing.test_routing",
])
def test_published_scenarios(module):
    if module.startswith(("bellium.cutout", "bellium.inpaint")):
        pytest.importorskip("PIL")
    importlib.import_module(module).run_tests()
