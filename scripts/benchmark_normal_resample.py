#!/usr/bin/env python3
"""Score the normal-field reductions and the drift a level-of-detail chain leaves.

Three published reductions are compared on controlled surfaces sampled at two
sizes, against the analytic field of the coarser sampling. The height column asks
the question a pipeline cares about: does the geometry survive the reduction.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bellium.material.controlled import sampled_geometry, sampled_height  # noqa: E402
from bellium.material.integration import height_error, integrate_height  # noqa: E402
from bellium.material.normals_io import inspect_normal_map  # noqa: E402
from bellium.material.photometric import normal_error  # noqa: E402
from bellium.material.resample import METHODS, downsample_normals, mip_chain  # noqa: E402

REPORT = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "manifests"
    / "normal-resample-v1.json"
)
GEOMETRIES = ("sphere", "cone", "waves", "tilted-plane")
BASE_SIZE = 32
FACTORS = (2, 4)
LEVELS = 2
# The wave family is the one whose content depends on the sampling: a twelve-pixel
# period is not band limited at sixteen pixels, so both periods are reported.
WAVE_PERIODS = (12.0, 24.0)


def scale_of(geometry: str, size: int) -> float:
    return 1.0 if geometry == "waves" else 1.0 / (size * 0.45)


def combined(left: list, right: list) -> list:
    rows = len(left)
    columns = len(left[0])
    return [
        [1.0 if (left[r][c] > 0.0 and right[r][c] > 0.0) else 0.0 for c in range(columns)]
        for r in range(rows)
    ]


def normalized(field: list) -> list:
    """The same directions with unit length, which isolates the direction error."""
    out = []
    for row in field:
        values = []
        for normal in row:
            if normal is None:
                values.append(None)
                continue
            length = math.sqrt(sum(value * value for value in normal))
            values.append(normal if length <= 1e-12 else tuple(v / length for v in normal))
        out.append(values)
    return out


def as_stored_bytes(field: list) -> list:
    """What a writer that does not normalize would put on disk, quantized to 8 bits."""
    return [
        [
            (128, 128, 255)
            if normal is None
            else tuple(
                min(255, max(0, int(round((value + 1.0) * 0.5 * 255.0)))) for value in normal
            )
            for normal in row
        ]
        for row in field
    ]


def reduction_section() -> dict:
    cases = []
    for geometry in GEOMETRIES:
        periods = WAVE_PERIODS if geometry == "waves" else (12.0,)
        for period in periods:
            normals, mask = sampled_geometry(geometry, BASE_SIZE, base_period=period)
            for factor in FACTORS:
                target = BASE_SIZE // factor
                truth, truth_mask = sampled_geometry(
                    geometry, target, base_period=period
                ), None
                truth_normals, truth_mask = truth
                truth_height, truth_height_mask = sampled_height(
                    geometry, target, base_period=period
                )
                for method in METHODS:
                    reduced = downsample_normals(
                        normals, factor, mask=mask, method=method
                    )
                    both = combined(reduced["mask"], truth_mask)
                    raw = normal_error(reduced["normals"], truth_normals, both)
                    direction = normal_error(
                        normalized(reduced["normals"]), truth_normals, both
                    )
                    result = integrate_height(
                        reduced["normals"],
                        reduced["mask"],
                        method="cumulative-average",
                        pixel_scale=scale_of(geometry, target),
                    )
                    height = height_error(
                        result["height"], truth_height, truth_height_mask
                    )
                    inspected = inspect_normal_map(as_stored_bytes(reduced["normals"]))
                    cases.append({
                        "geometry": geometry,
                        "wave_period": period if geometry == "waves" else None,
                        "factor": factor,
                        "method": method,
                        "raw_mean_deg": raw["mean_deg"],
                        "raw_p95_deg": raw["p95_deg"],
                        "direction_mean_deg": direction["mean_deg"],
                        "direction_max_deg": direction["max_deg"],
                        "height_relative_rmse": height["relative_rmse"],
                        "mean_length_before": reduced["mean_length_before"],
                        "mean_non_unit_of_the_stored_bytes": inspected["mean_non_unit"],
                        "inspector_verdict": inspected["verdict"],
                        "pixels": raw["pixels"],
                    })
    return {
        "note": (
            "Both sides are the same continuous surface: the wave period scales with the "
            "grid, and the comparison is against the analytic field of the coarser "
            "sampling, so the number mixes the filter with the sampling difference. That "
            "difference is negligible for the smooth families and large for a wave that is "
            "not band limited at the target size, which is why two periods are reported. "
            "raw_mean_deg is the error as stored, direction_mean_deg normalizes both sides "
            "first: the difference between them is the stored length and nothing else. The "
            "height column does not separate the methods, because a scale on a normal "
            "cancels in the slope and the integral is scale invariant; it prices the "
            "resolution instead."
        ),
        "cases": cases,
    }


def chain_section() -> dict:
    cases = []
    for geometry in GEOMETRIES:
        normals, mask = sampled_geometry(geometry, BASE_SIZE)
        chain = mip_chain(normals, mask=mask, levels=LEVELS, method="renormalized")
        rows = []
        for level in chain["pixels"]:
            size = chain["levels"][level["level"]]["shape"][0]
            truth_normals, truth_mask = sampled_geometry(geometry, size)
            truth_height, truth_height_mask = sampled_height(geometry, size)
            both = combined(level["mask"], truth_mask)
            error = normal_error(level["normals"], truth_normals, both)
            result = integrate_height(
                level["normals"], level["mask"], method="cumulative-average",
                pixel_scale=scale_of(geometry, size),
            )
            height = height_error(result["height"], truth_height, truth_height_mask)
            rows.append({
                "level": level["level"],
                "shape": chain["levels"][level["level"]]["shape"],
                "mean_deg": error["mean_deg"],
                "max_deg": error["max_deg"],
                "height_relative_rmse": height["relative_rmse"],
                "mean_z": chain["levels"][level["level"]]["mean_z"],
                "coverage": chain["levels"][level["level"]]["coverage"],
            })
        cases.append({
            "geometry": geometry,
            "method": chain["method"],
            "drift": chain["drift"],
            "warnings": chain["warnings"],
            "levels": rows,
        })
    return {
        "note": (
            "Renormalized reduction, thirty-two pixels down twice. The drift is the change "
            "in mean Z across the chain: averaging tilts a curved field towards the viewer, "
            "which is the flattening a mip chain shows."
        ),
        "flattening_limit": 0.03,
        "cases": cases,
    }


def detectability_section() -> dict:
    """Can the inspector see the defect a naive reduction leaves? Measured.

    The repository's own encoder normalizes every normal before it writes it, so the
    bytes a naive chain produces through that path are healed. A writer that does not
    normalize is measured here, because that is the one that stores the defect.
    """
    cases = []
    normals, mask = sampled_geometry("sphere", BASE_SIZE)
    for factor in FACTORS:
        naive = downsample_normals(normals, factor, mask=mask, method="naive")
        clean = downsample_normals(normals, factor, mask=mask, method="renormalized")
        for label, reduced in (("naive", naive), ("renormalized", clean)):
            inspected = inspect_normal_map(as_stored_bytes(reduced["normals"]))
            lengths = [
                math.sqrt(sum(value * value for value in reduced["normals"][r][c]))
                for r in range(len(reduced["normals"]))
                for c in range(len(reduced["normals"][0]))
                if reduced["normals"][r][c] is not None and reduced["mask"][r][c] > 0.0
            ]
            cases.append({
                "factor": factor,
                "method": label,
                "mean_stored_length": round(sum(lengths) / len(lengths), 6),
                "mean_non_unit": inspected["mean_non_unit"],
                "verdict": inspected["verdict"],
            })
    return {
        "note": (
            "The length defect a naive chain leaves is real but small, and the inspector's "
            "non-unit limit is 0.2: this section prices the gap between the defect and the "
            "check that is supposed to catch it. Two honest caveats. The repository's "
            "encoder normalizes every normal it writes, so a naive chain pushed through "
            "encode_normal_map arrives healed and this table measures a writer that does "
            "not normalize. And the angular gap in the reduction table between naive and "
            "renormalized is that length read through a formula that assumes unit vectors: "
            "two parallel vectors have no angle, whichever length they carry."
        ),
        "non_unit_limit": 0.2,
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default=str(REPORT), help="report path")
    options = parser.parse_args()
    report = {
        "schema": "bellium.normal-resample-benchmark/v1",
        "note": (
            "Controlled fields only. Nothing here is a measurement of a real texture, and "
            "no GPU filter or mipmap generator was involved."
        ),
        "base_size": BASE_SIZE,
        "reduction": reduction_section(),
        "chain": chain_section(),
        "detectability": detectability_section(),
    }
    destination = Path(options.report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for case in report["reduction"]["cases"]:
        period = "" if case["wave_period"] is None else f" period={case['wave_period']:g}"
        print(
            f"reduce x{case['factor']} {case['geometry']:13s}{period:12s} "
            f"{case['method']:13s} raw={case['raw_mean_deg']}deg "
            f"direction={case['direction_mean_deg']}deg "
            f"height={case['height_relative_rmse']} |n|={case['mean_length_before']} "
            f"inspector={case['inspector_verdict']}"
        )
    for case in report["chain"]["cases"]:
        levels = " | ".join(
            f"{level['shape'][0]}: {level['mean_deg']}deg h={level['height_relative_rmse']} "
            f"z={level['mean_z']}"
            for level in case["levels"]
        )
        print(f"chain {case['geometry']:13s} drift={case['drift']} {levels} {case['warnings']}")
    for case in report["detectability"]["cases"]:
        print(
            f"detectability x{case['factor']} {case['method']:13s} "
            f"stored_length={case['mean_stored_length']} non_unit={case['mean_non_unit']} "
            f"verdict={case['verdict']}"
        )
    print(f"wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
