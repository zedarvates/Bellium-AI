import pytest

from bellium.physics.references import (
    STANDARD_GRAVITY,
    hydrostatic_pressure,
    jump_apex_height,
    jump_apex_time,
    launch_speed_for_height,
    normal_gravity,
    standard_atmosphere,
    weight_force,
)
from bellium.physics.units import UnitError


def test_wgs84_normal_gravity_matches_published_values() -> None:
    assert normal_gravity(0.0).value == pytest.approx(9.7803253359, abs=1e-9)
    assert normal_gravity(90.0).value == pytest.approx(9.8321849378, abs=1e-6)
    assert normal_gravity(45.0).value == pytest.approx(9.8062, abs=1e-3)


def test_free_air_gradient_is_about_three_milligal_per_metre() -> None:
    surface = normal_gravity(45.0).value
    kilometre = normal_gravity(45.0, altitude_m=1000.0).value
    assert (surface - kilometre) == pytest.approx(3.086e-3, rel=2e-2)


def test_standard_atmosphere_matches_published_values() -> None:
    sea = standard_atmosphere(0.0)
    assert sea["temperature"].value == pytest.approx(288.15)
    assert sea["pressure"].value == pytest.approx(101325.0)
    assert sea["density"].value == pytest.approx(1.225, abs=1e-3)
    tropopause = standard_atmosphere(11000.0)
    assert tropopause["pressure"].value == pytest.approx(22632.0, rel=1e-4)
    assert tropopause["temperature"].value == pytest.approx(216.65)
    twenty = standard_atmosphere(20000.0)
    assert twenty["pressure"].value == pytest.approx(5474.9, rel=1e-3)


def test_hydrostatic_and_weight_are_consistent() -> None:
    assert hydrostatic_pressure(10.0, density_kg_m3=1000.0).value == pytest.approx(
        1000.0 * STANDARD_GRAVITY * 10.0
    )
    assert weight_force(80.0).value == pytest.approx(784.532)


def test_jump_kinematics_are_inverses() -> None:
    speed = launch_speed_for_height(2.0).value
    assert jump_apex_height(speed).value == pytest.approx(2.0, rel=1e-9)
    assert jump_apex_time(speed).value > 0.0
    assert jump_apex_height(5.0).value == pytest.approx(1.2746, abs=1e-4)


def test_estimates_carry_units_assumptions_and_provenance() -> None:
    estimate = normal_gravity(45.0, altitude_m=1200.0)
    payload = estimate.as_dict()
    assert payload["unit"] == "m/s2"
    assert payload["method"].startswith("reference:")
    assert payload["assumptions"] and payload["provenance"]
    assert payload["valid_range"]["altitude_m"] == [0.0, 20000.0]
    assert payload["uncertainty"]["kind"] == "model-domain"


def test_domain_limits_are_refused() -> None:
    with pytest.raises(UnitError, match="latitude_deg"):
        normal_gravity(91.0)
    with pytest.raises(UnitError, match="altitude_m"):
        normal_gravity(45.0, altitude_m=25000.0)
    with pytest.raises(UnitError, match="altitude_m"):
        standard_atmosphere(21000.0)
    with pytest.raises(UnitError, match="mass_kg"):
        weight_force(-1.0)
    with pytest.raises(UnitError, match="depth_m"):
        hydrostatic_pressure(-1.0)
    with pytest.raises(UnitError, match="launch_speed_m_s"):
        jump_apex_height(-1.0)
