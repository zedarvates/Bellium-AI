from bellium.specialists.physical import estimate_physical

GRAVITY = {"latitude_deg": 45.0, "altitude_m": 1200.0}


def test_reference_answers_and_cheaper_tiers_are_reported_beside_it() -> None:
    result = estimate_physical({"family": "gravity", "inputs": GRAVITY})
    assert result.abstained is False
    assert result.output["status"] == "reference"
    assert result.output["unit"] == "m/s2"
    assert result.output["method"].startswith("reference:")
    assert result.output["provisional"] is False
    methods = {alternative["method"] for alternative in result.output["alternatives"]}
    assert any("physical-case-retrieval" in method for method in methods)
    assert result.output["worst_alternative_deviation"] is not None


def test_without_the_reference_the_result_is_provisional() -> None:
    result = estimate_physical(
        {"family": "gravity", "inputs": GRAVITY, "reference_available": False}
    )
    assert result.output["status"] == "approximation"
    assert result.output["provisional"] is True
    assert result.output["declared_band"] is not None
    assert result.output["published_worst_case"] is not None


def test_no_available_method_abstains() -> None:
    result = estimate_physical({
        "family": "gravity",
        "inputs": GRAVITY,
        "reference_available": False,
        "allow_regressor": False,
        "allow_case_table": False,
    })
    assert result.abstained is True
    assert result.output["reason"] == "no_available_method"


def test_atmosphere_density_has_no_regressor_and_uses_the_table() -> None:
    result = estimate_physical({
        "family": "atmosphere-density",
        "inputs": {"altitude_m": 5000.0},
        "reference_available": False,
    })
    assert result.output["status"] == "approximation"
    assert result.output["method"].endswith("physical-case-retrieval:v0")


def test_out_of_domain_inputs_are_refused_by_the_reference() -> None:
    blocked = False
    try:
        estimate_physical({"family": "gravity", "inputs": {"latitude_deg": 45.0,
                                                           "altitude_m": 25000.0}})
    except ValueError:
        blocked = True
    assert blocked


def test_unknown_family_abstains() -> None:
    result = estimate_physical({"family": "magnetism", "inputs": {}})
    assert result.abstained is True
    assert result.output["reason"] == "unknown_family"


def test_missing_inputs_raise() -> None:
    blocked = False
    try:
        estimate_physical({"family": "gravity", "inputs": {"latitude_deg": 45.0}})
    except ValueError:
        blocked = True
    assert blocked


def test_hydrostatic_preview_matches_the_hand_computation() -> None:
    result = estimate_physical({
        "family": "hydrostatic", "inputs": {"depth_m": 10.0, "density_kg_m3": 1000.0}
    })
    assert result.output["status"] == "reference"
    assert 1000.0 * 9.80665 * 10.0 == round(result.output["value"], 6)
