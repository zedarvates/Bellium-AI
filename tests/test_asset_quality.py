from pathlib import Path

from bellium.knn.asset_quality import evaluate_asset, quality_status, record_verified


def _report(seed: int = 8, *, family: str = "mesh") -> dict:
    features = {
        "mesh": {"manifold": .9, "normals": .9, "prompt_alignment": seed / 10,
                 "scale": .8, "topology": .85, "uv": .8},
        "image": {"aesthetic": .8, "artifact_free": .9, "composition": .8,
                  "prompt_alignment": seed / 10, "technical": .9},
    }[family]
    checks = {"decodable": True, "license_verified": True, "manifest_verified": True}
    checks.update({"finite_geometry": True, "nonempty_geometry": True} if family == "mesh"
                  else {"dimensions_valid": True})
    return {"family": family, "sha256": f"{seed:064x}", "size_bytes": 1024 + seed,
            "checks": checks, "features": features}


def test_asset_quality_knn_gates_and_shadow_verdict(tmp_path: Path) -> None:
    incomplete = _report()
    del incomplete["checks"]["license_verified"]
    advice = evaluate_asset(incomplete, memory_root=tmp_path)
    assert advice.status == "incomplete"

    rejected = _report()
    rejected["checks"]["finite_geometry"] = False
    advice = evaluate_asset(rejected, memory_root=tmp_path)
    assert advice.status == "rule_fail" and advice.verdict == "FAIL"

    for seed in (7, 8):
        record_verified(_report(seed), memory_root=tmp_path, verdict="pass",
                        verified_by="tests:mesh-harness", evidence_refs=(f"fixture:{seed}",))
    record_verified(_report(8), memory_root=tmp_path, verdict="pass",
                    verified_by="tests:mesh-harness", evidence_refs=("fixture:duplicate",))
    duplicate_advice = evaluate_asset(_report(10), memory_root=tmp_path)
    assert duplicate_advice.status == "abstain" and duplicate_advice.neighbor_count == 2

    record_verified(_report(9), memory_root=tmp_path, verdict="pass",
                    verified_by="tests:mesh-harness", evidence_refs=("fixture:9",))
    suggestion = evaluate_asset(_report(10), memory_root=tmp_path)
    assert suggestion.status == "suggest" and suggestion.verdict == "PASS"
    assert len(suggestion.neighbors) == 3 and suggestion.shadow_only and not suggestion.acted

    image = evaluate_asset(_report(8, family="image"), memory_root=tmp_path)
    assert image.status == "abstain" and image.neighbor_count == 0

    blocked = False
    try:
        record_verified(_report(), memory_root=tmp_path, verdict="pass",
                        verified_by="model:self-report")
    except ValueError:
        blocked = True
    assert blocked

    status = quality_status(tmp_path)
    assert status["recorded_outcomes"] == 4
    assert status["verified_assets"] == 3
    assert status["families_ready"] == ["mesh"]
    assert status["activation_allowed"] is False

