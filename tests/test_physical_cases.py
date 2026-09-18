import pytest

from bellium.knn.physical_cases import load_cases, retrieve_physical_case
from bellium.physics import hydrostatic_pressure, normal_gravity, standard_atmosphere


def test_one_dimensional_family_interpolates_between_cases() -> None:
    result = retrieve_physical_case(
        {"family": "atmosphere-pressure", "inputs": {"altitude_m": 11250.0}}
    )
    assert result.abstained is False
    assert result.output["method"] == "knn-case-bracketing"
    assert result.output["bracket"]["low"]["input"] == 11000.0
    assert result.output["bracket"]["high"]["input"] == 11500.0
    expected = standard_atmosphere(11250.0)["pressure"].value
    assert result.output["value"] == pytest.approx(expected, rel=1e-3)


def test_grid_family_interpolates_bilinearly() -> None:
    result = retrieve_physical_case(
        {"family": "gravity", "inputs": {"latitude_deg": 45.0, "altitude_m": 1200.0}}
    )
    assert result.abstained is False
    assert result.output["method"] == "knn-case-bilinear"
    expected = normal_gravity(45.0, altitude_m=1200.0).value
    assert result.output["value"] == pytest.approx(expected, abs=1e-3)


def test_hydrostatic_family_is_bilinear_and_exact_on_the_grid() -> None:
    result = retrieve_physical_case(
        {"family": "hydrostatic", "inputs": {"depth_m": 47.5, "density_kg_m3": 1025.0}}
    )
    assert result.abstained is False
    expected = hydrostatic_pressure(47.5, density_kg_m3=1025.0).value
    assert result.output["value"] == pytest.approx(expected, rel=1e-6)


def test_out_of_domain_abstains() -> None:
    result = retrieve_physical_case(
        {"family": "gravity", "inputs": {"latitude_deg": 45.0, "altitude_m": 25000.0}}
    )
    assert result.abstained is True
    assert result.output["reason"] == "out_of_domain"
    assert result.output["value"] is None


def test_unknown_family_abstains() -> None:
    result = retrieve_physical_case({"family": "magnetism", "inputs": {}})
    assert result.abstained is True
    assert result.output["reason"] == "unknown_family"


def test_missing_input_is_not_coerced() -> None:
    blocked = False
    try:
        retrieve_physical_case({"family": "gravity", "inputs": {"latitude_deg": 45.0}})
    except ValueError:
        blocked = True
    assert blocked


def test_sparse_memory_refuses_to_interpolate() -> None:
    memory = {
        "family": "gravity",
        # The declared domain is wide, but the cases only cover the equator band,
        # so a query near the pole has no neighbour support.
        "domain": {"latitude_deg": [-90.0, 90.0], "altitude_m": [0.0, 100.0]},
        "output_unit": "m/s2",
        "reference_method": "reference:wgs84-somigliana+free-air",
        "items": [
            {"id": "a", "source": "fixture", "inputs": {"latitude_deg": -10.0, "altitude_m": 0.0},
             "output": 9.78, "vector": [-0.111, 0.0]},
            {"id": "b", "source": "fixture", "inputs": {"latitude_deg": 10.0, "altitude_m": 100.0},
             "output": 9.79, "vector": [0.111, 0.005]},
            {"id": "c", "source": "fixture", "inputs": {"latitude_deg": 0.0, "altitude_m": 50.0},
             "output": 9.785, "vector": [0.0, 0.0025]},
        ],
    }
    result = retrieve_physical_case(
        {"family": "gravity", "inputs": {"latitude_deg": 80.0, "altitude_m": 75.0}},
        memory=memory,
    )
    assert result.abstained is True
    assert result.output["reason"] == "insufficient_neighbor_support"


def test_cases_are_declared_as_reference_outputs() -> None:
    cases = load_cases("atmosphere-density")
    assert cases["reference_method"] == "reference:isa"
    assert "not field measurements" in cases["notes"]
    assert all(item["source"].startswith("reference-model:") for item in cases["items"])
