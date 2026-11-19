import hashlib
import json

import pytest

from scripts import benchmark_tool_router_training as benchmark


def test_boundary_escalation_uses_policy_before_any_memory():
    samples = benchmark.boundary_samples()
    assert len(samples) == 144
    assert {target for _, target in samples} == {0, 1, 2}
    assert benchmark.check_policy(samples) == {
        "escalation_cases": 128, "all_escalated_before_models": True,
    }
    assert benchmark.check_runtime_domain(samples) == {
        "unsupported_raw_inputs_abstained": 144,
        "non_veto_hybrid_inputs_abstained": 16,
    }


def test_failed_promotion_preserves_previous_candidate_and_published(tmp_path, monkeypatch):
    published = tmp_path / "published.json"
    candidate = tmp_path / "candidate.json"
    report = tmp_path / "report.json"
    published.write_text("previous published", encoding="utf-8")
    candidate.write_text("previous candidate", encoding="utf-8")
    monkeypatch.setattr(benchmark, "DEST", published)
    monkeypatch.setattr(benchmark, "benchmark", lambda: (
        {"promotion": {"passed": False, "reasons": ["class_0_recall_below_threshold"]}}, {}))
    before = hashlib.sha256(published.read_bytes()).hexdigest()
    assert benchmark.main(["--output", str(report), "--candidate-output", str(candidate)]) == 1
    assert candidate.read_text() == "previous candidate"
    assert hashlib.sha256(published.read_bytes()).hexdigest() == before
    assert not json.loads(report.read_text())["promotion"]["passed"]


@pytest.mark.parametrize("target", ["report", "candidate"])
def test_distributed_weights_cannot_be_used_as_output(tmp_path, monkeypatch, target):
    published = tmp_path / "published.json"
    published.write_text("preserve", encoding="utf-8")
    monkeypatch.setattr(benchmark, "DEST", published)
    monkeypatch.setattr(benchmark, "benchmark", lambda: pytest.fail("training should not start"))
    args = ["--output", str(published if target == "report" else tmp_path / "report.json")]
    if target == "candidate":
        args += ["--candidate-output", str(published)]
    with pytest.raises(SystemExit, match="2"):
        benchmark.main(args)
    assert published.read_text() == "preserve"


def test_report_and_candidate_paths_must_differ(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, "benchmark", lambda: pytest.fail("training should not start"))
    path = str(tmp_path / "same.json")
    with pytest.raises(SystemExit, match="2"):
        benchmark.main(["--output", path, "--candidate-output", path])
