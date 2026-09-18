import pytest

from bellium.physics.units import PhysicalEstimate, UnitError, convert, unit


def test_compatible_units_convert() -> None:
    assert convert(1.0, "km", "m") == pytest.approx(1000.0)
    assert convert(3600.0, "km/h", "m/s") == pytest.approx(1000.0)
    assert convert(1.0, "bar", "hPa") == pytest.approx(1000.0)
    assert convert(1.0, "mgal", "m/s2") == pytest.approx(1e-5)


def test_temperature_uses_an_affine_offset() -> None:
    assert convert(0.0, "degC", "K") == pytest.approx(273.15)
    assert convert(273.15, "K", "degC") == pytest.approx(0.0)


def test_dimension_mismatch_is_refused() -> None:
    with pytest.raises(UnitError, match="cannot convert"):
        convert(1.0, "m", "kg")


def test_unknown_unit_is_refused() -> None:
    with pytest.raises(UnitError, match="unknown unit"):
        unit("furlong")


def test_non_finite_value_is_refused() -> None:
    with pytest.raises(UnitError):
        convert(float("nan"), "m", "km")


def test_estimate_requires_provenance_and_a_known_unit() -> None:
    with pytest.raises(UnitError, match="provenance"):
        PhysicalEstimate(value=1.0, unit_name="m", method="reference:test")
    with pytest.raises(UnitError, match="unknown unit"):
        PhysicalEstimate(value=1.0, unit_name="banana", method="reference:test", provenance="x")


def test_estimate_converts_in_place_without_losing_metadata() -> None:
    estimate = PhysicalEstimate(
        value=9.80665,
        unit_name="m/s2",
        method="reference:test",
        assumptions=("constant gravity",),
        valid_range={"altitude_m": [0.0, 1000.0]},
        provenance="unit test",
        uncertainty={"kind": "model-domain"},
        inputs={"latitude_deg": 45.0},
    )
    converted = estimate.in_units("mgal")
    assert converted.value == pytest.approx(980665.0)
    assert converted.assumptions == estimate.assumptions
    assert converted.as_dict()["provenance"] == "unit test"
    assert converted.as_dict()["inputs"] == {"latitude_deg": 45.0}
