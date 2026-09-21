"""Deterministic physical references for game and simulation previews.

Every function returns a PhysicalEstimate carrying units, assumptions, declared
domain and provenance. These published models are the reference inside their
domain. No field measurement was compared, so the declared uncertainty is the
model domain and its documented approximation order, never a measured error.
"""

from __future__ import annotations

import math

from bellium.physics.units import PhysicalEstimate, UnitError

# WGS-84 ellipsoid constants, as published for the normal gravity formula.
WGS84_A_M = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = 0.00669437999014
WGS84_NORMAL_G_EQUATOR = 9.7803253359
WGS84_NORMAL_K = 0.00193185265241
WGS84_PROVENANCE = "WGS-84 normal gravity (Somigliana), published constants"

# Free-air reduction (Helmert form, mGal per metre, second order included).
FREE_AIR_LINEAR = 0.3087691
FREE_AIR_LATITUDE = 0.0004398
FREE_AIR_QUADRATIC = 7.2125e-8
FREE_AIR_PROVENANCE = "standard free-air reduction with second-order term"
MGAL = 1e-5

# International Standard Atmosphere.
ISA_SEA_LEVEL_TEMPERATURE_K = 288.15
ISA_LAPSE_RATE_K_PER_M = 0.0065
ISA_SEA_LEVEL_PRESSURE_PA = 101325.0
ISA_TROPOPAUSE_M = 11000.0
ISA_STRATOSPHERE_TEMPERATURE_K = 216.65
ISA_STRATOSPHERE_PRESSURE_PA = 22632.06
ISA_GAS_CONSTANT = 287.05287
ISA_EXPONENT = 5.2558797
ISA_PROVENANCE = "International Standard Atmosphere (ISA), troposphere and lower stratosphere"

STANDARD_GRAVITY = 9.80665
STANDARD_GRAVITY_PROVENANCE = "standard gravity g0, as used by the ISA definition"

ISA_VALID_ALTITUDE_M = 20000.0
FREE_AIR_VALID_ALTITUDE_M = 20000.0


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise UnitError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise UnitError(f"{name} must be finite")
    return number


def _latitude(value: object) -> float:
    latitude = _number(value, "latitude_deg")
    if not -90.0 <= latitude <= 90.0:
        raise UnitError("latitude_deg must be between -90 and 90")
    return latitude


def _altitude(value: object, limit: float) -> float:
    altitude = _number(value, "altitude_m")
    if not 0.0 <= altitude <= limit:
        raise UnitError(f"altitude_m must be between 0 and {limit:g}")
    return altitude


def normal_gravity(latitude_deg: object, *, altitude_m: object = 0.0) -> PhysicalEstimate:
    """Local gravitational acceleration from the WGS-84 model plus a free-air term."""
    latitude = _latitude(latitude_deg)
    altitude = _altitude(altitude_m, FREE_AIR_VALID_ALTITUDE_M)
    sin2 = math.sin(math.radians(latitude)) ** 2
    surface = (
        WGS84_NORMAL_G_EQUATOR
        * (1.0 + WGS84_NORMAL_K * sin2)
        / math.sqrt(1.0 - WGS84_E2 * sin2)
    )
    reduction = (
        -(FREE_AIR_LINEAR - FREE_AIR_LATITUDE * sin2) * altitude
        + FREE_AIR_QUADRATIC * altitude * altitude
    ) * MGAL
    return PhysicalEstimate(
        value=surface + reduction,
        unit_name="m/s2",
        method="reference:wgs84-somigliana+free-air",
        assumptions=(
            "WGS-84 ellipsoid, normal gravity at the ellipsoid surface",
            "altitude above the ellipsoid, not above mean sea level",
            "free-air reduction only: no terrain, no Bouguer or isostatic term",
        ),
        valid_range={"latitude_deg": [-90.0, 90.0], "altitude_m": [0.0, FREE_AIR_VALID_ALTITUDE_M]},
        provenance=f"{WGS84_PROVENANCE}; {FREE_AIR_PROVENANCE}",
        uncertainty={
            "kind": "model-domain",
            "note": "the free-air term is an approximation of the vertical gradient; "
                    "local mass anomalies are not modelled and are not measured here",
        },
        inputs={"latitude_deg": latitude, "altitude_m": altitude},
    )


def weight_force(mass_kg: object, gravity: object = STANDARD_GRAVITY) -> PhysicalEstimate:
    """Weight of a mass under a declared gravitational acceleration."""
    mass = _number(mass_kg, "mass_kg")
    if mass < 0.0:
        raise UnitError("mass_kg must not be negative")
    g = _number(gravity, "gravity")
    if g <= 0.0:
        raise UnitError("gravity must be positive")
    return PhysicalEstimate(
        value=mass * g,
        unit_name="N",
        method="reference:newton-weight",
        assumptions=("constant gravity over the body", "no buoyancy, no rotation term"),
        valid_range={"mass_kg": [0.0, None], "gravity_m_s2": [0.0, None]},
        provenance="definition of weight, F = m g",
        uncertainty={"kind": "input", "note": "the declared gravity carries its own domain"},
        inputs={"mass_kg": mass, "gravity_m_s2": g},
    )


def standard_atmosphere(altitude_m: object) -> dict[str, PhysicalEstimate]:
    """ISA temperature, pressure and density in the troposphere and lower stratosphere."""
    altitude = _altitude(altitude_m, ISA_VALID_ALTITUDE_M)
    if altitude <= ISA_TROPOPAUSE_M:
        temperature = ISA_SEA_LEVEL_TEMPERATURE_K - ISA_LAPSE_RATE_K_PER_M * altitude
        pressure = ISA_SEA_LEVEL_PRESSURE_PA * (
            temperature / ISA_SEA_LEVEL_TEMPERATURE_K
        ) ** ISA_EXPONENT
    else:
        temperature = ISA_STRATOSPHERE_TEMPERATURE_K
        pressure = ISA_STRATOSPHERE_PRESSURE_PA * math.exp(
            -STANDARD_GRAVITY * (altitude - ISA_TROPOPAUSE_M)
            / (ISA_GAS_CONSTANT * ISA_STRATOSPHERE_TEMPERATURE_K)
        )
    density = pressure / (ISA_GAS_CONSTANT * temperature)
    shared = {
        "method": "reference:isa",
        "assumptions": (
            "standard atmosphere: dry air, hydrostatic equilibrium",
            "no wind, no weather, no humidity",
        ),
        "valid_range": {"altitude_m": [0.0, ISA_VALID_ALTITUDE_M]},
        "provenance": ISA_PROVENANCE,
        "uncertainty": {
            "kind": "model-domain",
            "note": "a real day can differ by several percent; this is a standard model",
        },
    }
    return {
        "temperature": PhysicalEstimate(
            value=temperature, unit_name="K", inputs={"altitude_m": altitude}, **shared
        ),
        "pressure": PhysicalEstimate(
            value=pressure, unit_name="Pa", inputs={"altitude_m": altitude}, **shared
        ),
        "density": PhysicalEstimate(
            value=density, unit_name="kg/m3", inputs={"altitude_m": altitude}, **shared
        ),
    }


def hydrostatic_pressure(depth_m: object, *, density_kg_m3: object = 1025.0,
                        gravity: object = STANDARD_GRAVITY) -> PhysicalEstimate:
    """Gauge pressure of a fluid column: p = rho g h."""
    depth = _number(depth_m, "depth_m")
    if depth < 0.0:
        raise UnitError("depth_m must not be negative")
    density = _number(density_kg_m3, "density_kg_m3")
    if density <= 0.0:
        raise UnitError("density_kg_m3 must be positive")
    g = _number(gravity, "gravity")
    if g <= 0.0:
        raise UnitError("gravity must be positive")
    return PhysicalEstimate(
        value=density * g * depth,
        unit_name="Pa",
        method="reference:hydrostatic",
        assumptions=(
            "constant density with depth",
            "gauge pressure: atmospheric pressure on the surface is not added",
        ),
        valid_range={"depth_m": [0.0, None], "density_kg_m3": [0.0, None]},
        provenance="hydrostatic equilibrium for an incompressible column",
        uncertainty={"kind": "model-domain", "note": "real water density varies with salinity and temperature"},
        inputs={"depth_m": depth, "density_kg_m3": density, "gravity_m_s2": g},
    )


def dynamic_pressure(density_kg_m3: object, speed_m_s: object) -> PhysicalEstimate:
    """Dynamic pressure q = rho v^2 / 2. A drag force additionally needs a drag coefficient."""
    density = _number(density_kg_m3, "density_kg_m3")
    if density <= 0.0:
        raise UnitError("density_kg_m3 must be positive")
    speed = _number(speed_m_s, "speed_m_s")
    if speed < 0.0:
        raise UnitError("speed_m_s must not be negative")
    return PhysicalEstimate(
        value=0.5 * density * speed * speed,
        unit_name="Pa",
        method="reference:dynamic-pressure",
        assumptions=("incompressible flow", "no drag coefficient is assumed"),
        valid_range={"density_kg_m3": [0.0, None], "speed_m_s": [0.0, None]},
        provenance="definition of dynamic pressure",
        uncertainty={"kind": "input", "note": "a drag force needs a measured coefficient and is not estimated"},
        inputs={"density_kg_m3": density, "speed_m_s": speed},
    )


def jump_apex_height(launch_speed_m_s: object, *, gravity: object = STANDARD_GRAVITY) -> PhysicalEstimate:
    """Apex height of a vertical launch, ignoring drag: h = v^2 / (2 g)."""
    speed = _number(launch_speed_m_s, "launch_speed_m_s")
    if speed < 0.0:
        raise UnitError("launch_speed_m_s must not be negative")
    g = _number(gravity, "gravity")
    if g <= 0.0:
        raise UnitError("gravity must be positive")
    return PhysicalEstimate(
        value=speed * speed / (2.0 * g),
        unit_name="m",
        method="reference:ballistic-apex",
        assumptions=("no air drag", "flat ground", "constant gravity"),
        valid_range={"launch_speed_m_s": [0.0, None], "gravity_m_s2": [0.0, None]},
        provenance="constant-acceleration kinematics",
        uncertainty={"kind": "model-domain", "note": "drag and ground slope are not modelled"},
        inputs={"launch_speed_m_s": speed, "gravity_m_s2": g},
    )


def jump_apex_time(launch_speed_m_s: object, *, gravity: object = STANDARD_GRAVITY) -> PhysicalEstimate:
    """Time to the apex of a vertical launch: t = v / g."""
    speed = _number(launch_speed_m_s, "launch_speed_m_s")
    if speed < 0.0:
        raise UnitError("launch_speed_m_s must not be negative")
    g = _number(gravity, "gravity")
    if g <= 0.0:
        raise UnitError("gravity must be positive")
    return PhysicalEstimate(
        value=speed / g,
        unit_name="s",
        method="reference:ballistic-apex-time",
        assumptions=("no air drag", "constant gravity"),
        valid_range={"launch_speed_m_s": [0.0, None], "gravity_m_s2": [0.0, None]},
        provenance="constant-acceleration kinematics",
        uncertainty={"kind": "model-domain", "note": "drag is not modelled"},
        inputs={"launch_speed_m_s": speed, "gravity_m_s2": g},
    )


def launch_speed_for_height(target_height_m: object, *,
                            gravity: object = STANDARD_GRAVITY) -> PhysicalEstimate:
    """Required vertical launch speed for a target apex height: v = sqrt(2 g h)."""
    height = _number(target_height_m, "target_height_m")
    if height < 0.0:
        raise UnitError("target_height_m must not be negative")
    g = _number(gravity, "gravity")
    if g <= 0.0:
        raise UnitError("gravity must be positive")
    return PhysicalEstimate(
        value=math.sqrt(2.0 * g * height),
        unit_name="m/s",
        method="reference:launch-speed",
        assumptions=("no air drag", "constant gravity"),
        valid_range={"target_height_m": [0.0, None], "gravity_m_s2": [0.0, None]},
        provenance="constant-acceleration kinematics",
        uncertainty={"kind": "model-domain", "note": "drag is not modelled"},
        inputs={"target_height_m": height, "gravity_m_s2": g},
    )
