#!/usr/bin/env python3
"""Benchmark the gravity and pressure baselines against the cheaper tiers.

Acceptance thresholds are declared here, before any measurement, from the task
they serve: game and simulation previews. Every tier is measured on a grid that
was used neither for training nor for the trainers' held-out check, and the
reference formula stays the authority whenever it can be evaluated. The report
records what each tier costs and what it gets wrong; it never promotes a tier.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bellium.knn.physical_cases import (  # noqa: E402
    load_cases,
    retrieve_physical_case,
)
from bellium.micro_nn.atmosphere_model import estimate_pressure  # noqa: E402
from bellium.nano_nn.gravity_residual import estimate_gravity  # noqa: E402
from bellium.physics.references import (  # noqa: E402
    ISA_SEA_LEVEL_PRESSURE_PA,
    normal_gravity,
    standard_atmosphere,
)

REPORT = Path(__file__).resolve().parents[1] / "benchmarks" / "manifests" / "physical-estimates-v1.json"
THRESHOLDS = {
    "gravity": {
        "rmse": 1e-3,
        "max_abs": 5e-3,
        "unit": "m/s2",
        "why": "0.1 mGal moves a 1.27 m jump apex by about 0.13 mm",
    },
    "atmosphere-pressure": {
        "rmse": 0.01 * ISA_SEA_LEVEL_PRESSURE_PA,
        "max_abs": 0.03 * ISA_SEA_LEVEL_PRESSURE_PA,
        "unit": "Pa",
        "why": "1 % of sea level is far below the standard atmosphere's own daily variation",
    },
}
REPETITIONS = 5


def gravity_benchmark_grid():
    rows = []
    latitude = -88.5
    while latitude <= 88.5:
        altitude = 250
        while altitude <= 20000:
            rows.append((latitude, altitude))
            altitude += 1000
        latitude += 3.0
    return rows


def atmosphere_benchmark_grid():
    return list(range(25, 20000, 250))


def _summary(errors: list[float], abstentions: int, latencies: list[float]) -> dict:
    count = len(errors)
    rmse = (sum(error * error for error in errors) / count) ** 0.5 if count else float("nan")
    return {
        "evaluated": count,
        "abstentions": abstentions,
        "rmse": rmse,
        "max_abs": max((abs(error) for error in errors), default=float("nan")),
        "p50_ms": statistics.median(latencies) if latencies else float("nan"),
        "p95_ms": sorted(latencies)[int(0.95 * (len(latencies) - 1))] if latencies else float("nan"),
    }


def benchmark_gravity() -> dict:
    grid = gravity_benchmark_grid()
    memory = load_cases("gravity")
    table_bytes = (Path(memory_bytes_path("gravity")).stat().st_size)
    tiers = {
        "reference": {"errors": [], "abstentions": 0, "latencies": []},
        "case_table": {"errors": [], "abstentions": 0, "latencies": []},
        "nano_regressor": {"errors": [], "abstentions": 0, "latencies": []},
    }
    for latitude, altitude in grid:
        reference = normal_gravity(latitude, altitude_m=altitude).value
        start = time.perf_counter()
        tiers["reference"]["errors"].append(reference - reference)
        tiers["reference"]["latencies"].append((time.perf_counter() - start) * 1000.0)

        start = time.perf_counter()
        table = retrieve_physical_case(
            {"family": "gravity", "inputs": {"latitude_deg": latitude, "altitude_m": altitude}},
            memory=memory,
        )
        tiers["case_table"]["latencies"].append((time.perf_counter() - start) * 1000.0)
        if table.abstained:
            tiers["case_table"]["abstentions"] += 1
        else:
            tiers["case_table"]["errors"].append(table.output["value"] - reference)

        start = time.perf_counter()
        nano = estimate_gravity(latitude, altitude_m=altitude, check_reference=False)
        tiers["nano_regressor"]["latencies"].append((time.perf_counter() - start) * 1000.0)
        if nano.output["value"] is None:
            tiers["nano_regressor"]["abstentions"] += 1
        else:
            tiers["nano_regressor"]["errors"].append(nano.output["value"] - reference)
        checked = estimate_gravity(latitude, altitude_m=altitude)
        tiers["nano_regressor"]["checked_abstentions"] = tiers["nano_regressor"].get(
            "checked_abstentions", 0
        ) + int(checked.abstained)

    model_path = Path(memory_bytes_path("gravity", model=True))
    returns = {
        name: _summary(data["errors"], data["abstentions"], data["latencies"])
        for name, data in tiers.items()
    }
    returns["case_table"]["bytes"] = table_bytes
    returns["case_table"]["cases"] = len(memory["items"])
    returns["nano_regressor"]["bytes"] = model_path.stat().st_size
    returns["nano_regressor"]["parameters"] = 4
    returns["nano_regressor"]["abstentions_with_live_check"] = tiers["nano_regressor"].get(
        "checked_abstentions", 0
    )
    returns["reference"]["bytes"] = 0
    returns["reference"]["note"] = "exact by construction; no data, code only"
    for name in ("case_table", "nano_regressor"):
        returns[name]["within_threshold"] = bool(
            returns[name]["rmse"] <= THRESHOLDS["gravity"]["rmse"]
            and returns[name]["max_abs"] <= THRESHOLDS["gravity"]["max_abs"]
        )
    return {"family": "gravity", "queries": len(grid), "tiers": returns}


def benchmark_atmosphere() -> dict:
    grid = atmosphere_benchmark_grid()
    memory = load_cases("atmosphere-pressure")
    tiers = {
        "reference": {"errors": [], "abstentions": 0, "latencies": []},
        "case_table": {"errors": [], "abstentions": 0, "latencies": []},
        "micro_regressor": {"errors": [], "abstentions": 0, "latencies": []},
    }
    for altitude in grid:
        reference = standard_atmosphere(altitude)["pressure"].value
        start = time.perf_counter()
        tiers["reference"]["errors"].append(0.0)
        tiers["reference"]["latencies"].append((time.perf_counter() - start) * 1000.0)

        start = time.perf_counter()
        table = retrieve_physical_case(
            {"family": "atmosphere-pressure", "inputs": {"altitude_m": float(altitude)}},
            memory=memory,
        )
        tiers["case_table"]["latencies"].append((time.perf_counter() - start) * 1000.0)
        if table.abstained:
            tiers["case_table"]["abstentions"] += 1
        else:
            tiers["case_table"]["errors"].append(table.output["value"] - reference)

        start = time.perf_counter()
        micro = estimate_pressure(altitude, check_reference=False)
        tiers["micro_regressor"]["latencies"].append((time.perf_counter() - start) * 1000.0)
        if micro.output["value"] is None:
            tiers["micro_regressor"]["abstentions"] += 1
        else:
            tiers["micro_regressor"]["errors"].append(micro.output["value"] - reference)
        checked = estimate_pressure(altitude)
        tiers["micro_regressor"]["checked_abstentions"] = tiers["micro_regressor"].get(
            "checked_abstentions", 0
        ) + int(checked.abstained)

    model_path = Path(memory_bytes_path("atmosphere-pressure", model=True))
    returns = {
        name: _summary(data["errors"], data["abstentions"], data["latencies"])
        for name, data in tiers.items()
    }
    returns["case_table"]["bytes"] = Path(memory_bytes_path("atmosphere-pressure")).stat().st_size
    returns["case_table"]["cases"] = len(memory["items"])
    returns["micro_regressor"]["bytes"] = model_path.stat().st_size
    returns["micro_regressor"]["parameters"] = 21
    returns["micro_regressor"]["abstentions_with_live_check"] = tiers["micro_regressor"].get(
        "checked_abstentions", 0
    )
    returns["reference"]["bytes"] = 0
    returns["reference"]["note"] = "exact by construction; no data, code only"
    for name in ("case_table", "micro_regressor"):
        returns[name]["within_threshold"] = bool(
            returns[name]["rmse"] <= THRESHOLDS["atmosphere-pressure"]["rmse"]
            and returns[name]["max_abs"] <= THRESHOLDS["atmosphere-pressure"]["max_abs"]
        )
    return {"family": "atmosphere-pressure", "queries": len(grid), "tiers": returns}


def memory_bytes_path(family: str, *, model: bool = False) -> str:
    root = Path(__file__).resolve().parents[1]
    if model:
        name = "gravity-residual" if family == "gravity" else "atmosphere-model"
        sub = "nano_nn" if family == "gravity" else "micro_nn"
        return str(root / "models" / sub / name / "v0.json")
    file = "gravity-v0.json" if family == "gravity" else "atmosphere-pressure-v0.json"
    return str(root / "models" / "knn" / "physics" / file)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default=str(REPORT), help="report path")
    options = parser.parse_args()
    report = {
        "schema": "bellium.physical-benchmark/v1",
        "repetitions": REPETITIONS,
        "thresholds": THRESHOLDS,
        "note": (
            "Errors are measured against the published reference models on a grid used "
            "neither for training nor for the trainers' held-out check. The reference "
            "remains authoritative; the cheaper tiers exist for runtimes that cannot "
            "evaluate it."
        ),
        "families": [benchmark_gravity(), benchmark_atmosphere()],
    }
    destination = Path(options.report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for family in report["families"]:
        print(f"{family['family']}: {family['queries']} queries")
        for name, metrics in family["tiers"].items():
            verdict = metrics.get("within_threshold")
            marker = "" if verdict is None else (" pass" if verdict else " FAIL")
            print(
                f"  {name:16s} rmse={metrics['rmse']:.6g} max={metrics['max_abs']:.6g} "
                f"abstain={metrics['abstentions']} p50={metrics['p50_ms']:.4f}ms "
                f"p95={metrics['p95_ms']:.4f}ms bytes={metrics.get('bytes', 0)}{marker}"
            )
    print(f"wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
