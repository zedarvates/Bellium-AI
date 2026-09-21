"""Engine-import validation: declared contracts, findings and manifest mirrors."""

from copy import deepcopy
import json

import pytest

from bellium.imports.manifest import build_manifest, check_manifest, page_names
from bellium.imports.targets import (
    Finding,
    get_profile,
    is_power_of_two,
    load_profiles,
    summarize,
    target_profile,
    validate_plan,
)
from bellium.atlas.packing import pack, parse_frames
from bellium.specialists.imports import validate_import

FRAMES = [
    ("idle_0", 24, 32), ("idle_1", 24, 32), ("run_0", 28, 24), ("run_1", 28, 24),
    ("jump_0", 20, 40), ("hit_0", 36, 36), ("fx_0", 48, 12), ("icon_0", 12, 12),
]
PROFILE_KEYS = {
    "id": "test-square-pot",
    "source": "declared-contract:test",
    "note": "Test contract with power-of-two and square requirements.",
    "manifest_shape": "generic-regions",
    "color_space": "sRGB",
    "max_texture": 256,
    "power_of_two": True,
    "square": True,
    "required_padding": 2,
    "recommended_padding": 3,
    "max_pages_per_asset": 1,
}


def _frames(spec=FRAMES):
    return parse_frames([{"name": name, "width": w, "height": h} for name, w, h in spec])


def _codes(findings):
    return sorted(item.code for item in findings)


def _profile(**overrides):
    return target_profile({**PROFILE_KEYS, **overrides})


def test_shipped_profiles_load_with_provenance() -> None:
    profiles = load_profiles()
    ids = [profile.id for profile in profiles]
    assert ids == ["godot-4", "unity-sprite-atlas", "phaser-3-json", "generic-regions"]
    assert all(profile.source.startswith("declared-contract:") for profile in profiles)
    assert all(profile.note for profile in profiles)


def test_profile_lookup_refuses_an_unknown_id() -> None:
    with pytest.raises(KeyError):
        get_profile("godot-5")
    with pytest.raises(ValueError):
        get_profile(None)


@pytest.mark.parametrize("overrides", [
    {"id": ""},
    {"source": "  "},
    {"manifest_shape": "unity-meta"},
    {"color_space": "AdobeRGB"},
    {"max_texture": 0},
    {"max_texture": 1024.0},
    {"power_of_two": "yes"},
    {"square": 1},
    {"required_padding": -1},
    {"recommended_padding": 1},
    {"max_pages_per_asset": 0},
    {"unknown": 1},
])
def test_profile_contract_is_strict(overrides) -> None:
    with pytest.raises(ValueError):
        target_profile({**PROFILE_KEYS, **overrides})


def test_profile_requires_every_field() -> None:
    incomplete = dict(PROFILE_KEYS)
    incomplete.pop("max_texture")
    with pytest.raises(ValueError):
        target_profile(incomplete)


def test_power_of_two_helper() -> None:
    assert is_power_of_two(1) and is_power_of_two(1024) and not is_power_of_two(24)
    assert not is_power_of_two(0) and not is_power_of_two(-4)


def test_a_fitting_plan_reports_only_the_padding_warning() -> None:
    frames = _frames()
    plan = pack(frames, width=128, height=128, padding=2)
    assert plan.complete
    findings = validate_plan(frames, plan, _profile())
    assert _codes(findings) == ["padding_below_recommended"]
    assert findings[0].severity == "warning"
    assert summarize(findings) == {
        "errors": 0, "warnings": 1, "by_code": {"padding_below_recommended": 1},
    }


def test_required_padding_is_an_error() -> None:
    frames = _frames()
    plan = pack(frames, width=128, height=128)
    findings = validate_plan(frames, plan, _profile(recommended_padding=2))
    assert _codes(findings) == ["padding_below_required"]
    assert findings[0].severity == "error"


def test_unplaced_frames_block_the_import() -> None:
    frames = _frames()
    plan = pack(frames, width=32, height=32, max_pages=1)
    findings = validate_plan(frames, plan, _profile(required_padding=0, recommended_padding=0))
    codes = _codes(findings)
    assert codes.count("unplaced_frame") == len(plan.unplaced)
    assert len(plan.unplaced) > 0
    assert all(item.severity == "error" for item in findings if item.code == "unplaced_frame")
    names = {item.frame for item in findings if item.code == "unplaced_frame"}
    assert names == {item.name for item in plan.unplaced}


def test_page_limits_and_shape_requirements_fire() -> None:
    frames = _frames()
    wide = pack(frames, width=256, height=128, padding=4)
    findings = validate_plan(frames, wide, _profile())
    codes = _codes(findings)
    assert "not_square" in codes and "not_power_of_two" not in codes
    small = pack(frames, width=192, height=192, padding=4)
    findings = validate_plan(frames, small, _profile())
    assert "not_power_of_two" in _codes(findings)
    huge = pack(frames, width=512, height=512, padding=4)
    findings = validate_plan(frames, huge, _profile(power_of_two=False, square=False))
    assert "page_too_large" in _codes(findings)


def test_page_count_is_bounded_by_the_profile() -> None:
    frames = _frames([(f"q_{i}", 16, 16) for i in range(8)])
    plan = pack(frames, width=32, height=32, max_pages=4)
    assert plan.page_count == 2
    findings = validate_plan(frames, plan, _profile(power_of_two=False, square=False,
                                                   required_padding=0, recommended_padding=0))
    assert "too_many_pages" in _codes(findings)


def test_plan_invariants_are_reported_as_findings() -> None:
    frames = _frames()
    plan = pack(frames, width=128, height=128)
    broken = plan.__class__(
        pages=plan.pages, page_width=plan.page_width, page_height=plan.page_height,
        padding=plan.padding, method=plan.method, unplaced=plan.unplaced,
        used_area=plan.used_area + 1,
    )
    findings = validate_plan(frames, broken, _profile(required_padding=0, recommended_padding=0))
    assert any(item.code == "plan_invalid" and "used_area" in item.message for item in findings)


@pytest.mark.parametrize("shape", ["generic-regions", "godot-atlas-textures",
                                   "unity-spritesheet", "phaser-json"])
def test_every_shape_builds_a_verifiable_manifest(shape) -> None:
    frames = _frames()
    profile = _profile(manifest_shape=shape, required_padding=0, recommended_padding=0,
                       power_of_two=False, square=False)
    plan = pack(frames, width=128, height=128)
    manifest = build_manifest(frames, plan, profile)
    assert manifest["shape"] == shape
    assert manifest["engine_verified"] is False
    assert manifest["origin"] == ("bottom-left" if shape == "unity-spritesheet" else "top-left")
    assert check_manifest(manifest, frames, plan, profile) == []


def test_unity_rects_use_a_bottom_left_origin() -> None:
    frames = _frames()
    profile = _profile(manifest_shape="unity-spritesheet", required_padding=0,
                       recommended_padding=0, power_of_two=False, square=False)
    plan = pack(frames, width=128, height=128)
    manifest = build_manifest(frames, plan, profile)
    sprites = {sprite["name"]: sprite["rect"]
               for texture in manifest["textures"] for sprite in texture["sprites"]}
    for page in plan.pages:
        for placement in page:
            rect = sprites[placement.name]
            assert rect["x"] == placement.x
            assert rect["y"] == plan.page_height - (placement.y + placement.height)
            assert rect["w"] == placement.width and rect["h"] == placement.height


def test_phaser_multi_page_links_related_packs() -> None:
    frames = _frames([(f"q_{i}", 16, 16) for i in range(8)])
    profile = _profile(manifest_shape="phaser-json", required_padding=0, recommended_padding=0,
                       power_of_two=False, square=False, max_pages_per_asset=4)
    plan = pack(frames, width=32, height=32, max_pages=4)
    manifest = build_manifest(frames, plan, profile)
    assert manifest["meta"]["image"] == "atlas_00.png"
    assert manifest["meta"]["related_multi_packs"] == ["atlas_01.png"]
    assert {entry["page"] for entry in manifest["frames"].values()} == {0, 1}
    assert check_manifest(manifest, frames, plan, profile) == []


def test_page_names_are_rendered_and_validated() -> None:
    assert page_names(2) == ("atlas_00.png", "atlas_01.png")
    assert page_names(1, "sheet.png") == ("sheet.png",)
    assert page_names(2, "page_{index}.png") == ("page_0.png", "page_1.png")
    assert page_names(0) == ()
    with pytest.raises(ValueError):
        page_names(2, "sheet.png")  # a multi-page pattern needs the placeholder
    with pytest.raises(ValueError):
        page_names(2, "{name}.png")
    with pytest.raises(ValueError):
        page_names(-1)
    with pytest.raises(ValueError):
        page_names(1, "")


def test_manifest_tampering_is_caught() -> None:
    frames = _frames()
    profile = _profile(required_padding=0, recommended_padding=0, power_of_two=False, square=False)
    plan = pack(frames, width=128, height=128)
    manifest = build_manifest(frames, plan, profile)
    moved = deepcopy(manifest)
    moved["pages"][0]["regions"][0]["x"] += 1
    assert any("geometry differs" in item for item in check_manifest(moved, frames, plan, profile))
    flipped = deepcopy(manifest)
    flipped["pages"][0]["regions"][0]["y"] = plan.page_height - 1
    violations = check_manifest(flipped, frames, plan, profile)
    assert violations
    verified = deepcopy(manifest)
    verified["engine_verified"] = True
    assert any("engine_verified" in item for item in check_manifest(verified, frames, plan, profile))
    dropped = deepcopy(manifest)
    dropped["pages"][0]["regions"].pop()
    assert any("missing from the manifest" in item
               for item in check_manifest(dropped, frames, plan, profile))
    extra = deepcopy(manifest)
    extra["pages"][0]["regions"].append({"name": "ghost", "page": 0, "x": 0, "y": 0, "w": 1, "h": 1})
    assert any("no frame" in item for item in check_manifest(extra, frames, plan, profile))


def test_the_specialist_reports_a_ready_contract() -> None:
    query = {"frames": [{"name": n, "width": w, "height": h} for n, w, h in FRAMES],
             "width": 128, "height": 128, "padding": 2, "profile": "generic-regions"}
    before = deepcopy(query)
    result = validate_import(query)
    assert result.abstained is False
    assert result.output["status"] == "ready"
    assert result.output["profile"] == "generic-regions"
    assert result.output["manifest_shape"] == "generic-regions"
    assert result.output["manifest_violations"] == []
    assert result.output["imported"] is False
    assert result.output["written_files"] is False
    assert result.output["certified"] is False
    assert result.output["summary"]["errors"] == 0
    assert result.confidence == 0.9
    assert json.dumps(result.output)
    assert query == before


def test_the_specialist_abstains_on_a_contract_violation() -> None:
    frames = [{"name": n, "width": w, "height": h} for n, w, h in FRAMES]
    too_big = validate_import({"frames": frames, "width": 16384, "height": 16384,
                               "profile": "phaser-3-json"})
    assert too_big.abstained is True
    assert too_big.output["reason"] == "import_contract_violated"
    assert "page_too_large" in too_big.output["summary"]["by_code"]
    assert too_big.output["manifest_violations"] == []
    too_many = validate_import({"frames": frames, "width": 32, "height": 32,
                                "profile": "godot-4"})
    assert too_many.output["reason"] == "import_contract_violated"
    assert too_many.output["summary"]["by_code"].get("unplaced_frame")


def test_the_specialist_abstains_when_its_own_manifest_fails(monkeypatch) -> None:
    from bellium.specialists import imports as specialist

    monkeypatch.setattr(specialist, "build_manifest", lambda *args, **kwargs: {"shape": "broken"})
    result = specialist.validate_import({
        "frames": [{"name": n, "width": w, "height": h} for n, w, h in FRAMES],
        "width": 128, "height": 128, "profile": "generic-regions",
    })
    assert result.abstained is True
    assert result.output["reason"] == "manifest_failed_verification"
    assert result.output["manifest_violations"]


@pytest.mark.parametrize("query,error", [
    ({"frames": FRAMES, "width": 64, "height": 64}, ValueError),
    ({"frames": FRAMES, "width": 64, "height": 64, "profile": "nope"}, KeyError),
    ({"frames": FRAMES, "width": 64, "height": 64, "profile": "godot-4", "target": "godot-4"},
     ValueError),
    ({"frames": FRAMES, "width": 64, "height": 64, "profile": "godot-4", "page_pattern": 7},
     ValueError),
    ({"frames": FRAMES, "width": 64, "height": 64, "profile": "godot-4", "page_pattern": "x.png"},
     ValueError),
])
def test_the_specialist_refuses_invalid_queries(query, error) -> None:
    with pytest.raises(error):
        validate_import(query)


def test_the_specialist_refuses_a_non_object_query() -> None:
    with pytest.raises(ValueError):
        validate_import(["frames"])


def test_findings_are_frozen_values() -> None:
    finding = Finding("error", "code", "message")
    assert finding.frame is None
    with pytest.raises(Exception):
        finding.severity = "warning"

def test_profile_overrides_tighten_the_contract() -> None:
    frames = [{"name": n, "width": w, "height": h} for n, w, h in FRAMES]
    base = validate_import({"frames": frames, "width": 128, "height": 128,
                            "profile": "generic-regions"})
    assert base.abstained is False
    tightened = validate_import({"frames": frames, "width": 128, "height": 128,
                                 "profile": "generic-regions",
                                 "profile_overrides": {"max_texture": 48}})
    assert tightened.abstained is True
    assert "page_too_large" in tightened.output["summary"]["by_code"]
    assert tightened.output["manifest_violations"] == []
    same_id = validate_import({"frames": frames, "width": 128, "height": 128,
                               "profile": "generic-regions",
                               "profile_overrides": {"id": "generic-regions",
                                                     "recommended_padding": 4}})
    assert same_id.output["summary"]["by_code"] == {"padding_below_recommended": 1}


@pytest.mark.parametrize("overrides", [
    {"id": "other"}, {"unknown": 1}, {"max_texture": 0}, {"color_space": "AdobeRGB"},
    {"recommended_padding": -1}, "not-an-object", [], "", 0, None,
])
def test_profile_overrides_are_validated(overrides) -> None:
    with pytest.raises(ValueError):
        validate_import({
            "frames": [{"name": n, "width": w, "height": h} for n, w, h in FRAMES],
            "width": 128, "height": 128, "profile": "generic-regions",
            "profile_overrides": overrides,
        })
