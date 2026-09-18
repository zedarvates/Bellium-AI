import pytest

from bellium.material.controlled import (
    illumination_field,
    procedural_albedo,
    render,
    render_case,
)
from bellium.material.separation import (
    detail_energy,
    identity_separation,
    separate_albedo,
    separate_render,
    separation_error,
    unaffected_baseline,
)
from bellium.specialists.material import separate_image

SIZE = 24


def test_renders_are_deterministic() -> None:
    first = render_case("flat", "linear", size=SIZE, seed=5)
    second = render_case("flat", "linear", size=SIZE, seed=5)
    assert first["render"] == second["render"]
    assert first["render"] != render_case("flat", "linear", size=SIZE, seed=6)["render"]


def test_constant_illumination_means_the_render_is_the_albedo() -> None:
    albedo = procedural_albedo("checker", size=SIZE, seed=1)
    rendered = render(albedo, illumination_field("constant", size=SIZE))
    error = separation_error(identity_separation(rendered), albedo)
    assert error["rmse"] == pytest.approx(0.0, abs=1e-9)


def test_separation_wins_on_a_detailed_albedo_under_smooth_shading() -> None:
    case = render_case("stripes", "linear", size=SIZE, seed=2)
    identity = unaffected_baseline(case["render"], case["albedo"])
    separated = separation_error(separate_albedo(case["render"]), case["albedo"])
    assert separated["rmse"] < identity["rmse"]


def test_a_flat_albedo_is_flagged_as_not_identifiable() -> None:
    # Which method wins on a flat albedo depends on size and radius, so the
    # benchmark scores that; here the guarantee is the warning itself.
    from bellium.material.controlled import to_image

    case = render_case("flat", "vignette", size=SIZE, seed=2)
    result = separate_image({"image": to_image(case["render"]), "radius": 4})
    assert "flat_albedo_not_identifiable" in result.output["warnings"]


def test_declared_prior_determines_the_answer() -> None:
    case = render_case("stripes", "linear", size=SIZE, seed=3)
    smooth = separate_render(case["render"], expected_shading="smooth")
    none = separate_render(case["render"], expected_shading="none")
    assert smooth["choice"] == "prior:smooth" and smooth["unresolved"] is False
    assert none["choice"] == "prior:none" and none["unresolved"] is False
    assert smooth["method"] != none["method"]
    assert smooth["albedo"] != none["albedo"]


def test_without_a_prior_both_candidates_are_reported() -> None:
    case = render_case("checker", "linear", size=SIZE, seed=4)
    report = separate_render(case["render"])
    assert report["unresolved"] is True
    assert report["choice"] == "recommended"
    assert set(report["candidates"]) == {"identity", "frequency-separation"}
    for key in ("detail_energy", "low_frequency_repeats", "shading_variation", "clipped_ratio"):
        assert key in report["evidence"]


def test_detail_energy_is_larger_for_a_structured_albedo() -> None:
    structured = render_case("checker", "constant", size=SIZE, seed=7)["render"]
    smooth = render_case("gradient", "constant", size=SIZE, seed=7)["render"]
    assert detail_energy(structured) > detail_energy(smooth)


def test_specialist_reports_the_prior_and_its_warnings() -> None:
    case = render_case("checker", "vignette", size=SIZE, seed=8)
    from bellium.material.controlled import to_image

    result = separate_image({
        "image": to_image(case["render"]),
        "expected_shading": "smooth",
        "radius": 4,
    })
    assert result.output["status"] == "ready"
    assert result.output["method"] == "log-homomorphic-separation"
    assert result.output["certified"] is False
    assert result.output["pixels_changed"] == 0
    assert "shading_prior_required" not in result.output["warnings"]


def test_specialist_requires_a_prior_when_none_is_declared() -> None:
    case = render_case("checker", "vignette", size=SIZE, seed=8)
    from bellium.material.controlled import to_image

    result = separate_image({"image": to_image(case["render"]), "radius": 4})
    assert result.output["unresolved"] is True
    assert "shading_prior_required" in result.output["warnings"]


def test_specialist_abstains_on_clipped_highlights() -> None:
    blown = [[(255, 255, 255) for _ in range(SIZE)] for _ in range(SIZE)]
    result = separate_image({"image": blown, "expected_shading": "smooth"})
    assert result.abstained is True
    assert "highlights_clipped" in result.output["warnings"]


def test_invalid_inputs_are_refused() -> None:
    case = render_case("checker", "linear", size=SIZE, seed=9)
    from bellium.material.controlled import to_image

    image = to_image(case["render"])
    for query in (
        {"image": image, "expected_shading": "guessed"},
        {"image": image, "radius": 0},
        {"image": image, "gamma": -1.0},
    ):
        blocked = False
        try:
            separate_image(query)
        except ValueError:
            blocked = True
        assert blocked


def test_out_of_range_channels_are_refused() -> None:
    blocked = False
    try:
        separate_render([[(-0.1, 0.0, 0.0), (0.5, 0.5, 0.5)]])
    except ValueError:
        blocked = True
    assert blocked
