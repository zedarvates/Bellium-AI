# P10 drafting gate: 2D document, exact primitives, SVG/DXF round-trip

The gate asks for a native 2D drawing before any CAD operator or pixel trace.
Every case is hand-authored geometry, so the measurement is reproducible without
scans, photographs or a modelling kernel.

    python scripts/benchmark_drafting.py

Report: [benchmarks/manifests/drafting-roundtrip-v1.json](../benchmarks/manifests/drafting-roundtrip-v1.json).

## What exists

| Piece | Behaviour |
| --- | --- |
| document | Y-up, origin at the paper bottom-left, coordinates in mm or in |
| layers | unique ASCII names; every entity names a declared layer |
| primitives | line, polyline, circle, arc, text |
| SVG profile | data-bellium drafting-v0; model space stays Y-up |
| DXF R12 | ASCII LINE, CIRCLE, ARC, TEXT, POLYLINE; INSUNITS and LIMMAX |
| inspect_drawing | consultative specialist; emits SVG and DXF only after both round-trips match |

This is not raster-to-svg. Pixel contours stay in the existing raster-to-svg specialist.
Offset, trim, cotation, blocks and 3D CAD are out of this gate.

## Invariants, recomputed on every fixture

| Invariant | Result |
| --- | --- |
| Unknown keys, non-finite numbers and degenerate geometry raise | holds |
| Unsupported kinds (spline, DXF SPLINE, foreign SVG) abstain | holds |
| Empty entity list abstains | holds |
| SVG round-trip matches the canonical document | holds |
| DXF round-trip matches the canonical document, including unit and paper | holds |
| Y-up model coordinates are not flipped by the SVG profile | holds |
| pixels_written stays false; authority stays consultative | holds |

## Authored fixtures

| Name | Unit | Paper | Contents |
| --- | --- | --- | --- |
| plate-a4 | mm | 210 x 297 | rectangle, diagonal, title text |
| washer | mm | 100 x 100 | two concentric circles on two layers |
| bracket | mm | 80 x 70 | closed L-polyline |
| slot | mm | 100 x 80 | two lines and two 180 degree arcs |
| inch-card | in | 6 x 4 | rectangle, circle, title text |

No fixture is a traced scan. No neural tier is retained: the document is the
baseline, and a model would only hide a failed round-trip.

## Measured result

5 fixtures x SVG and DXF, 20 repeats, 0 invariant violations, 0 mismatches,
0 specialist failures. Round-trip latency p50 0.07 ms, p95 0.11 ms on this machine.

| Fixture | Unit | Entities | SVG bytes | DXF bytes |
| --- | --- | ---: | ---: | ---: |
| plate-a4 | mm | 6 | 1036 | 567 |
| washer | mm | 2 | 539 | 370 |
| bracket | mm | 1 | 430 | 494 |
| slot | mm | 4 | 793 | 430 |
| inch-card | in | 6 | 1044 | 568 |
