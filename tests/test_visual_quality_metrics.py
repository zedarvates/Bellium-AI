"""The quality gate must not hide image errors behind unmasked pixels or abstention."""
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/evaluate_visual_quality.py"
spec = importlib.util.spec_from_file_location("visual_quality", SCRIPT)
quality = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quality)


def test_metric_measures_only_missing_pixels_but_checks_untouched_ones():
    original = [[(100, 100, 100), (50, 50, 50)]]
    changed = [[(0, 0, 0), (51, 50, 50)]]
    result = quality.mask_metrics(original, changed, [[1, 0]])
    assert result["mae_255"] == 100.0
    assert result["known_pixels_preserved"] is False


def test_baselines_do_not_use_hidden_pixel_values():
    mask = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    left = [[(80, 120, 160)] * 3 for _ in range(3)]
    right = [row[:] for row in left]
    left[1][1] = (0, 0, 0)
    right[1][1] = (255, 0, 255)
    assert quality.nearest_known(left, mask) == quality.nearest_known(right, mask)
    assert quality.ring_mean(left, mask) == quality.ring_mean(right, mask)


def test_abstaining_on_every_case_does_not_pass_quality():
    limits = json.loads((SCRIPT.parent.parent / "benchmarks/manifests/visual-quality-v1.json").read_text())["inpaint"]["gates"]
    cases = [{"abstained": True, "elapsed_seconds": 0.1, "patch": {"known_pixels_preserved": True}}]
    result = quality.inpaint_summary(cases, limits)
    assert result["passed"] is False
    assert result["coverage"] == 0
    assert result["accepted_mean_mae_255"] is None


def test_cached_dataset_mismatch_is_not_silently_replaced(tmp_path):
    path = tmp_path / "image.png"
    path.write_bytes(b"modified input")
    protocol = {"upstream_revision": "a" * 40, "assets": [{"file": "image.png", "sha256": "b" * 64}]}
    with pytest.raises(ValueError, match="hash mismatch"):
        quality.fetch_assets(protocol, tmp_path, fetch=False)
    assert path.read_bytes() == b"modified input"
