"""Unit-safe scalars for physical previews.

Conversions are explicit and dimension-checked. An incompatible dimension
raises instead of returning a plausible number, and temperatures use an
explicit affine offset rather than a bare factor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


class UnitError(ValueError):
    """Raised when units are unknown or incompatible."""


@dataclass(frozen=True)
class Unit:
    name: str
    dimension: str
    to_base: float
    offset: float = 0.0
    base: str = ""

    def to_base_value(self, value: float) -> float:
        return value * self.to_base + self.offset

    def from_base_value(self, value: float) -> float:
        return (value - self.offset) / self.to_base


def _unit(name: str, dimension: str, to_base: float, base: str, *, offset: float = 0.0) -> Unit:
    return Unit(name, dimension, to_base, offset, base)


UNITS: dict[str, Unit] = {
    item.name: item
    for item in (
        _unit("m", "length", 1.0, "m"),
        _unit("km", "length", 1000.0, "m"),
        _unit("cm", "length", 0.01, "m"),
        _unit("s", "time", 1.0, "s"),
        _unit("ms", "time", 0.001, "s"),
        _unit("kg", "mass", 1.0, "kg"),
        _unit("g", "mass", 0.001, "kg"),
        _unit("m/s", "speed", 1.0, "m/s"),
        _unit("km/h", "speed", 1000.0 / 3600.0, "m/s"),
        _unit("kn", "speed", 1852.0 / 3600.0, "m/s"),
        _unit("m/s2", "acceleration", 1.0, "m/s2"),
        _unit("gal", "acceleration", 0.01, "m/s2"),
        _unit("mgal", "acceleration", 1e-5, "m/s2"),
        _unit("N", "force", 1.0, "N"),
        _unit("kN", "force", 1000.0, "N"),
        _unit("Pa", "pressure", 1.0, "Pa"),
        _unit("hPa", "pressure", 100.0, "Pa"),
        _unit("kPa", "pressure", 1000.0, "Pa"),
        _unit("bar", "pressure", 100000.0, "Pa"),
        _unit("K", "temperature", 1.0, "K"),
        _unit("degC", "temperature", 1.0, "K", offset=273.15),
        _unit("kg/m3", "density", 1.0, "kg/m3"),
        _unit("deg", "angle", math.pi / 180.0, "rad"),
        _unit("rad", "angle", 1.0, "rad"),
    )
}


def unit(name: str) -> Unit:
    if not isinstance(name, str) or name not in UNITS:
        known = ", ".join(sorted(UNITS))
        raise UnitError(f"unknown unit '{name}'; known units: {known}")
    return UNITS[name]


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise UnitError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise UnitError(f"{name} must be finite")
    return number


def convert(value: object, from_unit: str, to_unit: str) -> float:
    """Convert between compatible units, raising on a dimension mismatch."""
    source = unit(from_unit)
    target = unit(to_unit)
    if source.dimension != target.dimension:
        raise UnitError(
            f"cannot convert {source.dimension} to {target.dimension} "
            f"({from_unit} -> {to_unit})"
        )
    return target.from_base_value(source.to_base_value(_finite(value, "value")))


@dataclass(frozen=True)
class PhysicalEstimate:
    """One estimate with its units, assumptions, domain and provenance."""

    value: float
    unit_name: str
    method: str
    assumptions: tuple[str, ...] = ()
    valid_range: dict[str, Any] = field(default_factory=dict)
    provenance: str = ""
    uncertainty: dict[str, Any] = field(default_factory=dict)
    inputs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _finite(self.value, "value"))
        if not isinstance(self.unit_name, str) or self.unit_name not in UNITS:
            raise UnitError(f"unknown unit '{self.unit_name}'")
        if not isinstance(self.method, str) or not self.method.strip():
            raise UnitError("method must be declared")
        if not isinstance(self.assumptions, tuple):
            raise UnitError("assumptions must be a tuple of strings")
        if not self.provenance:
            raise UnitError("provenance must be declared")

    def in_units(self, target: str) -> "PhysicalEstimate":
        converted = convert(self.value, self.unit_name, target)
        return PhysicalEstimate(
            value=converted,
            unit_name=target,
            method=self.method,
            assumptions=self.assumptions,
            valid_range=dict(self.valid_range),
            provenance=self.provenance,
            uncertainty=dict(self.uncertainty),
            inputs=dict(self.inputs),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "unit": self.unit_name,
            "method": self.method,
            "assumptions": list(self.assumptions),
            "valid_range": dict(self.valid_range),
            "provenance": self.provenance,
            "uncertainty": dict(self.uncertainty),
            "inputs": dict(self.inputs),
        }
