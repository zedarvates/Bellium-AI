#!/usr/bin/env python3
"""Generate the physical case tables from the published reference models.

The cases are model outputs, not field measurements. Each file declares the
domain in which interpolation is allowed and the reference method it came from.
Existing files are preserved unless --force is passed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bellium.physics import hydrostatic_pressure, normal_gravity, standard_atmosphere  # noqa: E402

DEST = Path(__file__).resolve().parents[1] / "models" / "knn" / "physics"
SCHEMA = "bellium.physical-case-memory/v1"
NOTES = (
    "Cases generated from published reference models by scripts/build_physical_cases.py. "
    "They are not field measurements; interpolation is limited to the declared domain."
)


def _grid(values: list[float]) -> list[float]:
    return [round(value, 6) for value in values]


def gravity_cases() -> dict:
    items = []
    for latitude in range(-90, 91, 15):
        for altitude in range(0, 20001, 2500):
            estimate = normal_gravity(latitude, altitude_m=altitude)
            items.append({
                "id": f"grav_{latitude:+04d}_{altitude:05d}",
                "inputs": {"latitude_deg": float(latitude), "altitude_m": float(altitude)},
                "output": round(estimate.value, 6),
                "source": "reference-model:wgs84-somigliana+free-air",
            })
    return {
        "schema": SCHEMA,
        "family": "gravity",
        "domain": {"latitude_deg": [-90.0, 90.0], "altitude_m": [0.0, 20000.0]},
        "output_unit": "m/s2",
        "reference_method": "reference:wgs84-somigliana+free-air",
        "notes": NOTES,
        "items": items,
    }


def atmosphere_cases(field: str, family: str, unit: str) -> dict:
    items = []
    for altitude in range(0, 20001, 500):
        estimate = standard_atmosphere(altitude)[field]
        items.append({
            "id": f"atm_{family.split('-')[-1]}_{altitude:05d}",
            "inputs": {"altitude_m": float(altitude)},
            "output": round(estimate.value, 8),
            "source": "reference-model:isa",
        })
    return {
        "schema": SCHEMA,
        "family": family,
        "domain": {"altitude_m": [0.0, 20000.0]},
        "output_unit": unit,
        "reference_method": "reference:isa",
        "notes": NOTES,
        "items": items,
    }


def hydrostatic_cases() -> dict:
    items = []
    for depth in range(0, 101, 5):
        for density in (1000.0, 1025.0, 1050.0):
            estimate = hydrostatic_pressure(depth, density_kg_m3=density)
            items.append({
                "id": f"hyd_{depth:03d}_{int(density)}",
                "inputs": {"depth_m": float(depth), "density_kg_m3": density},
                "output": round(estimate.value, 6),
                "source": "reference-model:hydrostatic",
            })
    return {
        "schema": SCHEMA,
        "family": "hydrostatic",
        "domain": {"depth_m": [0.0, 100.0], "density_kg_m3": [1000.0, 1050.0]},
        "output_unit": "Pa",
        "reference_method": "reference:hydrostatic",
        "notes": NOTES,
        "items": items,
    }


def write(path: Path, payload: dict, *, force: bool) -> str:
    if path.exists() and not force:
        return f"kept {path.name}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return f"wrote {path.name} ({len(payload['items'])} cases)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="overwrite existing tables")
    options = parser.parse_args()
    print(write(DEST / "gravity-v0.json", gravity_cases(), force=options.force))
    print(write(DEST / "atmosphere-pressure-v0.json",
                atmosphere_cases("pressure", "atmosphere-pressure", "Pa"), force=options.force))
    print(write(DEST / "atmosphere-density-v0.json",
                atmosphere_cases("density", "atmosphere-density", "kg/m3"), force=options.force))
    print(write(DEST / "hydrostatic-v0.json", hydrostatic_cases(), force=options.force))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
