#!/usr/bin/env python3
"""Score albedo and illumination separation against controlled renders.

Every case carries the maps that generated it, so both candidates and the
prior-declared answer can be scored exactly. The report also measures how often
an image-only recommendation picks the winning method, because that is the part
the image alone cannot settle.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bellium.material.controlled import render_case  # noqa: E402
from bellium.material.separation import (  # noqa: E402
    separate_albedo,
    separate_render,
    separation_error,
    unaffected_baseline,
)

REPORT = Path(__file__).resolve().parents[1] / "benchmarks" / "manifests" / "material-separation-v1.json"
ALBEDOS = ("flat", "gradient", "checker", "stripes", "color-blocks")
ILLUMINATIONS = ("constant", "linear", "vignette", "lamp", "shadow-band")
SEEDS = (3, 11)
SIZE = 32


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default=str(REPORT), help="report path")
    options = parser.parse_args()
    summary: dict = {
        "schema": "bellium.material-separation-benchmark/v1",
        "note": (
            "Controlled synthetic renders with known maps. Nothing here is a "
            "photograph and nothing about real captures is established."
        ),
        "size": SIZE,
        "seeds": list(SEEDS),
        "levels": {},
        "cases": [],
    }
    for albedo_kind in ALBEDOS:
        for illumination_kind in ILLUMINATIONS:
            for seed in SEEDS:
                case = render_case(
                    albedo_kind, illumination_kind, size=SIZE, seed=seed
                )
                truth = case["albedo"]
                start = time.perf_counter()
                identity = unaffected_baseline(case["render"], truth)
                separated = separation_error(separate_albedo(case["render"]), truth)
                prior = "none" if illumination_kind == "constant" else "smooth"
                report = separate_render(case["render"], expected_shading=prior)
                declared = separation_error(
                    {"albedo": report["albedo"], "illumination": report["illumination"]},
                    truth,
                )
                automatic = separate_render(case["render"])
                automatic_error = separation_error(
                    {"albedo": automatic["albedo"], "illumination": automatic["illumination"]},
                    truth,
                )
                elapsed = (time.perf_counter() - start) * 1000.0
                winner = "identity" if identity["rmse"] <= separated["rmse"] else "frequency-separation"
                record = {
                    "name": f"{case['name']}/seed={seed}",
                    "albedo_kind": albedo_kind,
                    "illumination_kind": illumination_kind,
                    "identity_rmse": identity["rmse"],
                    "separation_rmse": separated["rmse"],
                    "declared_rmse": declared["rmse"],
                    "automatic_rmse": automatic_error["rmse"],
                    "declared_prior": prior,
                    "winner": winner,
                    "recommendation": automatic["recommendation"],
                    "recommendation_hit": automatic["recommendation"] == winner,
                    "evidence": automatic["evidence"],
                    "latency_ms": elapsed,
                }
                summary["cases"].append(record)
                level = summary["levels"].setdefault(illumination_kind, {
                    "cases": 0, "identity_rmse": [], "separation_rmse": [],
                    "declared_rmse": [], "automatic_rmse": [], "wins": 0,
                    "recommendation_hits": 0, "latencies": [],
                })
                level["cases"] += 1
                level["identity_rmse"].append(identity["rmse"])
                level["separation_rmse"].append(separated["rmse"])
                level["declared_rmse"].append(declared["rmse"])
                level["automatic_rmse"].append(automatic_error["rmse"])
                level["wins"] += int(winner == "frequency-separation")
                level["recommendation_hits"] += int(record["recommendation_hit"])
                level["latencies"].append(elapsed)
    for level in summary["levels"].values():
        latencies = level.pop("latencies")
        for key in ("identity_rmse", "separation_rmse", "declared_rmse", "automatic_rmse"):
            values = level.pop(key)
            level[f"mean_{key}"] = round(statistics.mean(values), 6)
        level["p50_ms"] = round(statistics.median(latencies), 4)
    destination = Path(options.report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for name, level in summary["levels"].items():
        print(
            f"{name:12s} cases={level['cases']:2d} identity={level['mean_identity_rmse']:.4f} "
            f"separation={level['mean_separation_rmse']:.4f} "
            f"declared={level['mean_declared_rmse']:.4f} "
            f"automatic={level['mean_automatic_rmse']:.4f} "
            f"separation_wins={level['wins']}/{level['cases']} "
            f"recommendation_hits={level['recommendation_hits']}/{level['cases']} "
            f"p50={level['p50_ms']}ms"
        )
    print(f"wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
