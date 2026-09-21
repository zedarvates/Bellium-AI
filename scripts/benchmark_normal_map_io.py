#!/usr/bin/env python3
"""Score the normal-map encoding, the declared handedness and the symptoms.

The round trip says what 8 bits cost, the integrability check says what no
integrator can recover, the symptom table says which defects a single map can
decide, and the handedness table says which one it cannot. Nothing here is a
measurement of a real asset: the fields come from the controlled geometries.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bellium.material.controlled import perturbed_normals  # noqa: E402
from bellium.material.integration import (  # noqa: E402
    analytic_height,
    gradients,
    height_error,
    integrate_height,
    integrability,
)
from bellium.material.normals_io import (  # noqa: E402
    detect_convention,
    decode_normal_map,
    encode_normal_map,
    flip_normal_convention,
    inspect_normal_map,
    quantize_height,
)
from bellium.material.photometric import normal_error, synthetic_geometry  # noqa: E402

REPORT = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "manifests"
    / "normal-map-io-v1.json"
)
GEOMETRIES = ("sphere", "cone", "waves", "tilted-plane")
SIZE = 32
RADIUS = SIZE * 0.45
SIGMAS = (0.0, 0.05)
DISPLACEMENT_BITS = (8, 16)
ROTATIONAL_STRENGTH = 0.3


def scale_of(geometry: str) -> float:
    return 1.0 if geometry == "waves" else 1.0 / RADIUS


def rotational(slope_x, slope_y, valid, strength=ROTATIONAL_STRENGTH):
    rows, columns = len(valid), len(valid[0])
    points = [(r, c) for r in range(rows) for c in range(columns) if valid[r][c]]
    center_r = sum(r for r, _ in points) / len(points)
    center_c = sum(c for _, c in points) / len(points)
    out_x = [
        [slope_x[r][c] - (strength * (r - center_r) if valid[r][c] else 0.0)
         for c in range(columns)]
        for r in range(rows)
    ]
    out_y = [
        [slope_y[r][c] + (strength * (c - center_c) if valid[r][c] else 0.0)
         for c in range(columns)]
        for r in range(rows)
    ]
    return out_x, out_y


def round_trip_section() -> dict:
    cases = []
    for geometry in GEOMETRIES:
        normals, mask = synthetic_geometry(geometry, size=SIZE)
        for sigma in SIGMAS:
            field = perturbed_normals(normals, sigma, seed=3)
            image = encode_normal_map(field, convention="+y")
            reconstructed, report = decode_normal_map(
                image, convention="+y", z_from_xy=True, mask=mask
            )
            stored, _ = decode_normal_map(
                image, convention="+y", z_from_xy=False, mask=mask
            )
            cases.append({
                "geometry": geometry,
                "sigma": sigma,
                "z_from_xy": normal_error(reconstructed, field, mask),
                "stored_z": normal_error(stored, field, mask),
                "unreconstructable": report["unreconstructable"],
                "pixels": report["pixels"],
                "mean_non_unit": report["mean_non_unit"],
            })
    return {
        "note": (
            "8-bit RGB, n = 2c - 1, no sRGB transfer. z_from_xy reconstructs the blue "
            "channel from the first two; stored_z reads it back as written."
        ),
        "cases": cases,
    }


def integrability_section() -> dict:
    cases = []
    for geometry in GEOMETRIES:
        normals, mask = synthetic_geometry(geometry, size=SIZE)
        truth, truth_mask = analytic_height(geometry, size=SIZE)
        for sigma in (0.0, 0.02, 0.05, 0.1):
            field = perturbed_normals(normals, sigma, seed=2)
            slope_x, slope_y, valid = gradients(field, mask)
            clean = integrability(slope_x, slope_y, valid)
            result = integrate_height(
                field, mask, method="cumulative-average", pixel_scale=scale_of(geometry)
            )
            error = height_error(result["height"], truth, truth_mask)
            turned_x, turned_y = rotational(slope_x, slope_y, valid)
            turned = integrability(turned_x, turned_y, valid)
            cases.append({
                "geometry": geometry,
                "sigma": sigma,
                "curl_ratio": clean["curl_ratio"],
                "verdict": clean["verdict"],
                "cells": clean["cells"],
                "relative_rmse": error["relative_rmse"],
                "rotational_curl_ratio": turned["curl_ratio"],
                "rotational_verdict": turned["verdict"],
            })
    return {
        "note": (
            "The curl is measured on interior pixels only, with central differences. It "
            "states what no integrator can recover; it is not a prediction of the error "
            "of a given integrator, which the table shows."
        ),
        "limit": 0.05,
        "suspect_limit": 0.2,
        "cases": cases,
    }


def height_map_as_normal_map() -> tuple[list, list]:
    """A grayscale height map stored in RGB, read as if it were a normal map."""
    normals, mask = synthetic_geometry("sphere", size=SIZE)
    result = integrate_height(normals, mask, method="cumulative-average")
    values = [
        result["height"][r][c]
        for r in range(SIZE)
        for c in range(SIZE)
        if result["valid"][r][c]
    ]
    low, high = min(values), max(values)
    image = [[(0, 0, 0)] * SIZE for _ in range(SIZE)]
    for r in range(SIZE):
        for c in range(SIZE):
            if not result["valid"][r][c]:
                continue
            level = int(round((result["height"][r][c] - low) / (high - low) * 255.0))
            image[r][c] = (level, level, level)
    return image, result["valid"]


def object_space_field() -> list:
    """Every direction once, which is what an object-space map of a closed mesh carries."""
    rows = []
    for r in range(SIZE):
        row = []
        for c in range(SIZE):
            theta = math.pi * (c + 0.5) / SIZE
            phi = 2.0 * math.pi * (r + 0.5) / SIZE
            row.append((
                math.sin(theta) * math.cos(phi),
                math.sin(theta) * math.sin(phi),
                math.cos(theta),
            ))
        rows.append(row)
    return rows


def symptoms_section() -> dict:
    normals, mask = synthetic_geometry("sphere", size=SIZE)
    tangent = encode_normal_map(normals, convention="+y")
    inverted_z = encode_normal_map(
        [[(n[0], n[1], -n[2]) if n else None for n in row] for row in normals],
        convention="+y",
    )
    height_image, height_valid = height_map_as_normal_map()
    cases = []
    variants = (
        ("tangent-space", tangent, "+y", mask),
        ("flipped-y declared correctly", flip_normal_convention(tangent), "-y", mask),
        ("flipped-y read as +y", flip_normal_convention(tangent), "+y", mask),
        ("inverted-z", inverted_z, "+y", mask),
        ("object-space", encode_normal_map(object_space_field()), "+y", None),
        ("height map in RGB", height_image, "+y", height_valid),
    )
    for label, image, declaration, variant_mask in variants:
        report = inspect_normal_map(image, convention=declaration, mask=variant_mask)
        cases.append({
            "variant": label,
            "verdict": report["verdict"],
            "mean_z": report["mean_z"],
            "negative_z_ratio": report["negative_z_ratio"],
            "mean_non_unit": report["mean_non_unit"],
            "saturation_ratio": report["saturation_ratio"],
            "y_convention_decidable": report["y_convention_decidable"],
            "detection": report["y_convention_detection"]["status"],
            "detected": report["y_convention_detection"]["convention"],
        })
    return {
        "note": (
            "One controlled field per variant. The last two rows are the defects a caller "
            "actually meets: an object-space map and a height map stored in an image."
        ),
        "cases": cases,
    }


def detection_section() -> dict:
    """Can a map decide its own handedness? Measured, both answers counted."""
    cases = []
    decided = abstained = wrong = 0
    for geometry in GEOMETRIES:
        normals, mask = synthetic_geometry(geometry, size=SIZE)
        for sigma in (0.0, 0.02, 0.05, 0.1):
            field = perturbed_normals(normals, sigma, seed=4)
            for truth_convention in ("+y", "-y"):
                picture = encode_normal_map(field, convention=truth_convention)
                found = detect_convention(picture, mask=mask)
                correct = found["convention"] == truth_convention
                if found["status"] == "decided":
                    decided += 1
                    if not correct:
                        wrong += 1
                else:
                    abstained += 1
                cases.append({
                    "geometry": geometry,
                    "sigma": sigma,
                    "declared_in_the_picture": truth_convention,
                    "status": found["status"],
                    "reason": found["reason"],
                    "detected": found["convention"],
                    "correct": correct if found["status"] == "decided" else None,
                    "margin": found["margin"],
                    "curl_y_plus": found["readings"]["+y"]["curl_ratio"],
                    "curl_y_minus": found["readings"]["-y"]["curl_ratio"],
                })
    return {
        "note": (
            "Both handedness of the same controlled field, at four noise levels. A decided "
            "answer that is wrong is counted as wrong; an abstention is not a failure."
        ),
        "margin_limit": 2.0,
        "summary": {
            "runs": len(cases),
            "decided": decided,
            "abstained": abstained,
            "wrong": wrong,
        },
        "cases": cases,
    }


def displacement_section() -> dict:
    cases = []
    for geometry in ("sphere", "waves"):
        normals, mask = synthetic_geometry(geometry, size=SIZE)
        result = integrate_height(
            normals, mask, method="cumulative-average", pixel_scale=scale_of(geometry)
        )
        for depth in DISPLACEMENT_BITS:
            quantized = quantize_height(result["height"], result["valid"], bit_depth=depth)
            span = quantized["span"][1] - quantized["span"][0]
            cases.append({
                "geometry": geometry,
                "bit_depth": depth,
                "span": quantized["span"],
                "step": quantized["step"],
                "worst_error": quantized["worst_error"],
                "worst_error_of_span": round(quantized["worst_error"] / span, 8) if span else None,
                "pixels": quantized["pixels"],
            })
    return {
        "note": (
            "Displacement is returned as levels plus the declared span; no file is written, "
            "because the shipped PNG writer is 8-bit RGB and RGBA only."
        ),
        "cases": cases,
    }


def _srgb(value: float) -> float:
    return 12.92 * value if value <= 0.0031308 else 1.055 * value ** (1.0 / 2.4) - 0.055


def colour_space_section() -> dict:
    """What it costs to store a normal map as if it were colour, and to read it back."""
    cases = []
    for geometry in GEOMETRIES:
        normals, mask = synthetic_geometry(geometry, size=SIZE)
        image = encode_normal_map(normals)
        stored_as_colour = [
            [
                tuple(
                    min(255, max(0, int(round(_srgb(channel / 255.0) * 255.0))))
                    for channel in pixel
                )
                for pixel in row
            ]
            for row in image
        ]
        read_back, _ = decode_normal_map(stored_as_colour, mask=mask)
        cases.append({
            "geometry": geometry,
            "stored_as_colour_read_as_data": normal_error(read_back, normals, mask),
        })
    return {
        "note": (
            "The measured mapping is n = 2c - 1 with no transfer function. Applying the sRGB "
            "curve on the way in and reading the bytes back as data is the pipeline mistake "
            "this row prices."
        ),
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default=str(REPORT), help="report path")
    options = parser.parse_args()
    report = {
        "schema": "bellium.normal-map-io-benchmark/v1",
        "note": (
            "Controlled fields only. The handedness is declared by the caller; this report "
            "measures what a map can and cannot say about itself."
        ),
        "size": SIZE,
        "round_trip": round_trip_section(),
        "integrability": integrability_section(),
        "symptoms": symptoms_section(),
        "detection": detection_section(),
        "displacement": displacement_section(),
        "colour_space": colour_space_section(),
    }
    destination = Path(options.report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for case in report["round_trip"]["cases"]:
        print(
            f"round trip {case['geometry']:13s} sigma={case['sigma']:<5} "
            f"reconstructed mean={case['z_from_xy']['mean_deg']}deg "
            f"worst={case['z_from_xy']['max_deg']}deg "
            f"stored mean={case['stored_z']['mean_deg']}deg"
        )
    for case in report["integrability"]["cases"]:
        print(
            f"curl       {case['geometry']:13s} sigma={case['sigma']:<5} "
            f"{case['curl_ratio']} {case['verdict']:<14s} "
            f"rotational={case['rotational_curl_ratio']}"
        )
    for case in report["symptoms"]["cases"]:
        print(
            f"verdict    {case['variant']:28s} {str(case['verdict']):16s} "
            f"mean_z={case['mean_z']} non_unit={case['mean_non_unit']} "
            f"sat={case['saturation_ratio']}"
        )
    detection = report["detection"]
    for case in detection["cases"]:
        print(
            f"detection  {case['geometry']:13s} sigma={case['sigma']:<5} "
            f"picture={case['declared_in_the_picture']} "
            f"curl+={case['curl_y_plus']} curl-={case['curl_y_minus']} "
            f"margin={case['margin']} {case['status']:<8s} "
            f"detected={case['detected']} correct={case['correct']}"
        )
    summary = detection["summary"]
    print(
        f"detection summary: {summary['runs']} runs, {summary['decided']} decided, "
        f"{summary['abstained']} abstained, {summary['wrong']} wrong"
    )
    for case in report["displacement"]["cases"]:
        print(
            f"displacement {case['geometry']:8s} {case['bit_depth']:2d}-bit "
            f"step={case['step']} worst={case['worst_error']} "
            f"({case['worst_error_of_span']:.6%} of span)"
        )
    for case in report["colour_space"]["cases"]:
        error = case["stored_as_colour_read_as_data"]
        print(
            f"colour space {case['geometry']:13s} sRGB-in/linear-out mean={error['mean_deg']}deg "
            f"worst={error['max_deg']}deg"
        )
    print(f"wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
