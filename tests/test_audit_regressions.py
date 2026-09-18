"""Counterexamples from the 2026-09-15 audit: these must fail closed."""
import json
from copy import deepcopy
from dataclasses import replace

import pytest

from bellium.contracts import AuthorityMode, SpecialistResult
from bellium.knn.asset_quality import evaluate_asset
from bellium.knn.patch_inpaint import inpaint
from bellium.knn.phoneme import load_inventory, retrieve_phoneme
from bellium.knn.pronunciation import compare_pronunciation
from bellium.knn.tool_router import load_exemplars, retrieve_tool
from bellium.knn.visual_anomaly import assess_visual, load_exemplars as load_visual
from bellium.micro_nn.features import featurize
from bellium.specialists import tool_router


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), True, "0.5"])
def test_result_rejects_invalid_confidence(value):
    with pytest.raises(ValueError):
        SpecialistResult("audit", {}, value, False, AuthorityMode.OBSERVE)


def test_string_active_cannot_bypass_authority_validation():
    with pytest.raises(ValueError):
        SpecialistResult("audit", {}, 0.5, True, "active")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_micro_nn_rejects_nonfinite_features(value):
    with pytest.raises(ValueError):
        featurize("binary_router", {"complexity": value, "budget_ratio": 0.5, "has_local_model": 1.0})


@pytest.mark.parametrize("side", [3, 4, 5])
def test_inpaint_completes_inner_pixels_and_preserves_input(side):
    image = [[(200, 200, 200)] * 24 for _ in range(24)]
    mask = [[int(7 <= r < 7+side and 7 <= c < 7+side) for c in range(24)] for r in range(24)]
    for r in range(24):
        for c in range(24):
            if mask[r][c]:
                image[r][c] = (0, 0, 0)
    original, original_mask = deepcopy(image), deepcopy(mask)
    result = inpaint(image, mask)
    assert not result.abstained
    assert result.output["filled"] == side * side
    assert all(pixel == (200, 200, 200) for row in result.output["image"] for pixel in row)
    assert image == original and mask == original_mask


def test_patch_matching_ignores_the_unknown_center():
    image = [[(255, 255, 255)] * 24 for _ in range(24)]
    mask = [[0] * 24 for _ in range(24)]
    mask[12][12] = 1
    image[12][12] = (0, 0, 0)
    # Two candidate centers have identical known context; their centers must not
    # influence the patch distance, since the query center is unknown.
    from bellium.knn.patch_inpaint import _known_patch_distance
    image[9][9] = (0, 0, 0)
    assert _known_patch_distance(image, mask, 12, 12, image, 9, 9) == 0.0
    assert _known_patch_distance(image, mask, 12, 12, image, 15, 15) == 0.0


def test_inventory_rejects_a_different_header_language(tmp_path):
    path = tmp_path / "phones.json"
    path.write_text(json.dumps({"schema": "bellium.phoneme-inventory/v1", "language_id": "wrong-language",
                                "items": load_inventory()}), encoding="utf-8")
    with pytest.raises(ValueError, match="language"):
        load_inventory(path)


def test_equal_nearest_phones_are_ambiguous_in_either_order():
    memory = load_inventory()
    query = {"language_id": "fixture-lab", "features": {"voicing": 0.5, "manner": 0., "place": 0., "rounding": 0.}}
    for records in (memory, list(reversed(memory))):
        result = retrieve_phoneme(query, inventory=records)
        assert result.abstained
        assert result.output["certified"] is False


@pytest.mark.parametrize("phones", [["p", "p", "a", "t", "a"], ["p", "a", "a", "t", "a"], ["p", "a", "t"]])
def test_insertions_and_deletions_are_not_exact_pronunciations(phones):
    result = compare_pronunciation({"language_id": "fixture-lab", "form": "pata", "ipa": phones})
    assert not result.output["certified"]
    assert result.output["bucket"] != "match"
    assert result.output["mismatches"]


def _task():
    return {"features": {"has_code": 1., "has_files": 1., "has_error": 0., "wants_mutation": 0.,
                         "wants_external_state": 0., "criticality": 0.32},
            "signals": {"mentions_secret": False, "mentions_deploy": False, "mentions_destruction": False}}


def test_veto_does_not_load_a_model_or_require_features(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("model consulted after deterministic veto")
    monkeypatch.setattr(tool_router, "load_mlp", forbidden)
    result = tool_router.route_tool({"signals": {"mentions_secret": True}})
    assert result.output["tool"] == "escalate"
    assert result.output["micro_nn"] is None


def test_hybrid_abstains_when_nn_abstains(monkeypatch):
    baseline = tool_router.classify_tool_need(_task())
    monkeypatch.setattr(tool_router, "classify_tool_need", lambda _: replace(baseline, output={"label": None}, abstained=True))
    result = tool_router.route_tool(_task())
    assert result.abstained and result.output["tool"] is None


def test_fractional_mutation_uses_the_same_policy_threshold():
    query = _task()
    query["features"]["wants_mutation"] = 0.6
    result = retrieve_tool(query)
    assert result.output["status"] == "rule_escalate"


def test_repeated_tool_example_cannot_create_three_neighbors():
    item = load_exemplars()[0]
    query = {"features": item["features"]}
    result = retrieve_tool(query, memory=[deepcopy(item) for _ in range(3)])
    assert result.abstained
    assert len(result.output["neighbors"]) == 1


def test_bad_asset_ledger_lines_do_not_crash_or_create_support(tmp_path):
    (tmp_path / "asset-quality.jsonl").write_text('[]\nnull\n{"broken":true}\ninvalid json\n', encoding="utf-8")
    report = {"family": "mesh", "sha256": "a" * 64, "size_bytes": 1024,
              "features": {key: 0.8 for key in ("manifold", "normals", "prompt_alignment", "scale", "topology", "uv")},
              "checks": {key: True for key in ("decodable", "license_verified", "manifest_verified", "finite_geometry", "nonempty_geometry")}}
    result = evaluate_asset(report, memory_root=tmp_path)
    assert result.status == "abstain" and result.neighbor_count == 0


def test_mixed_visual_labels_abstain_even_outside_known_space():
    memory = [m for m in load_visual() if m["domain"] == "aquaponics-lab"]
    memory[0]["label"] = "novel"
    result = assess_visual({"domain": "aquaponics-lab", "image": [[(190, 24, 18)] * 12 for _ in range(12)]}, memory=memory)
    assert result.abstained and result.output["label"] is None
