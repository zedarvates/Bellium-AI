"""Engine-import validation: measured findings and manifest sizes per profile.

Five authored atlas cases run against the shipped profiles, plus one caller
override that tightens the contract. The report counts errors, warnings and
manifest bytes; it never claims an engine imported anything.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bellium.imports.targets import load_profiles  # noqa: E402
from bellium.specialists.imports import validate_import  # noqa: E402

REPORT = Path(__file__).resolve().parents[1] / "benchmarks" / "manifests" / "engine-import-v1.json"
REPEATS = 5
MEASURED = (
    ("f_0", 16, 12), ("f_1", 32, 24), ("f_2", 8, 40), ("f_3", 24, 16), ("f_4", 32, 40),
    ("f_5", 8, 32), ("f_6", 40, 16), ("f_7", 24, 16), ("f_8", 12, 16), ("f_9", 40, 12),
    ("f_10", 24, 24), ("f_11", 40, 24), ("f_12", 16, 32), ("f_13", 12, 24),
)
MIXED = (
    ("idle_0", 24, 32), ("idle_1", 24, 32), ("run_0", 28, 24), ("run_1", 28, 24),
    ("jump_0", 20, 40), ("jump_1", 20, 40), ("hit_0", 36, 36), ("fall_0", 16, 48),
    ("fall_1", 16, 48), ("fx_0", 48, 12), ("fx_1", 48, 12), ("icon_0", 12, 12),
)


def cases() -> list[dict]:
    return [
        {"name": "uniform_16", "frames": [(f"u_{i}", 16, 16) for i in range(64)],
         "width": 128, "height": 128, "max_pages": 1},
        {"name": "mixed_sprite", "frames": list(MIXED), "width": 128, "height": 128,
         "max_pages": 1},
        {"name": "mixed_sprite_padded", "frames": list(MIXED), "width": 128, "height": 128,
         "max_pages": 1, "padding": 2},
        {"name": "measured_96", "frames": list(MEASURED), "width": 96, "height": 96,
         "max_pages": 1},
        {"name": "three_pages", "frames": [(f"p_{i}", 32, 32) for i in range(24)],
         "width": 128, "height": 128, "max_pages": 3},
    ]


def _query(case: dict, profile_id: str, overrides: dict | None) -> dict:
    query = {
        "frames": [{"name": name, "width": width, "height": height}
                   for name, width, height in case["frames"]],
        "width": case["width"], "height": case["height"], "max_pages": case["max_pages"],
        "padding": case.get("padding", 0), "profile": profile_id,
    }
    if overrides:
        query["profile_overrides"] = overrides
    return query


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default=str(REPORT), help="report path")
    parser.add_argument("--repeats", type=int, default=REPEATS, help="timed repetitions")
    options = parser.parse_args()
    profiles = load_profiles()
    plan: list[tuple[str, str, dict | None]] = [
        (case["name"], profile.id, None) for case in cases() for profile in profiles
    ]
    plan.append(("measured_96", "generic-regions",
                 {"power_of_two": True, "square": True, "max_texture": 48,
                  "required_padding": 3, "recommended_padding": 3}))
    report: dict = {
        "schema": "bellium.engine-import-benchmark/v1",
        "note": (
            "Profiles are declared contracts with provenance, not measurements of an engine. "
            "A ready verdict means the geometry satisfies the contract; no engine ran and no "
            "file was written."
        ),
        "repeats": options.repeats,
        "runs": [],
    }
    by_code: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    manifest_bytes: list[int] = []
    violations_total = 0
    print(f"{'case':<22}{'profile':<22}{'status':<10}{'err':>4}{'warn':>5}{'bytes':>8}{'ms':>9}")
    for name, profile_id, overrides in plan:
        case = next(item for item in cases() if item["name"] == name)
        durations = []
        result = None
        for _ in range(max(1, options.repeats)):
            start = time.perf_counter()
            result = validate_import(_query(case, profile_id, overrides))
            durations.append((time.perf_counter() - start) * 1000.0)
        assert result is not None
        output = result.output
        size = len(json.dumps(output["manifest"], separators=(",", ":")))
        summary = output["summary"]
        violations_total += len(output["manifest_violations"])
        manifest_bytes.append(size)
        status_counts[output["status"]] = status_counts.get(output["status"], 0) + 1
        for code, count in summary["by_code"].items():
            by_code[code] = by_code.get(code, 0) + count
        report["runs"].append({
            "case": name,
            "profile": profile_id,
            "overridden": bool(overrides),
            "status": output["status"],
            "reason": output["reason"],
            "manifest_shape": output["manifest_shape"],
            "errors": summary["errors"],
            "warnings": summary["warnings"],
            "by_code": summary["by_code"],
            "manifest_bytes": size,
            "manifest_violations": output["manifest_violations"],
            "ms_p50": round(statistics.median(durations), 4),
            "imported": output["imported"],
            "written_files": output["written_files"],
            "certified": output["certified"],
        })
        print(f"{name:<22}{profile_id:<22}{output['status']:<10}"
              f"{summary['errors']:>4}{summary['warnings']:>5}{size:>8}"
              f"{statistics.median(durations):>9.3f}")
    report["summary"] = {
        "runs": len(report["runs"]),
        "status_counts": status_counts,
        "findings_by_code": by_code,
        "manifest_bytes_min": min(manifest_bytes),
        "manifest_bytes_max": max(manifest_bytes),
        "manifest_violations": violations_total,
        "certified_true": sum(1 for run in report["runs"] if run["certified"]),
        "imported_true": sum(1 for run in report["runs"] if run["imported"]),
    }
    target = Path(options.report)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("summary:", json.dumps(report["summary"]))
    print("report:", target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
