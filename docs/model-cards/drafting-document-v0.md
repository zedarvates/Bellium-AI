Model card: bellium/deterministic/drafting-document:v0

- Task: hold a 2D technical drawing as a native document (units, paper, layers,
  line / polyline / circle / arc / text) and emit SVG plus DXF R12 that round-trip
  back to the same document.
- Authority: consultative. pixels_written stays false. Nothing is a CAD edit.
- Public API: bellium.drafting.document.drawing_from_dict, check_drawing,
  drawings_match; drawing_to_svg / drawing_from_svg; drawing_to_dxf /
  drawing_from_dxf; the specialist is inspect_drawing with exactly one of
  drawing, svg or dxf.
- Method: deterministic parse, invariant check, then dual export. The specialist
  refuses to emit when either round-trip fails.
- Coordinate frame: Y-up, origin at the paper bottom-left. SVG carries
  data-y=up and a flip transform for display only; import reads model
  attributes, not screen space.
- Units: mm or in. DXF stores INSUNITS (4 or 1) and paper size in LIMMAX.
- Bounds: no spline, ellipse, hatch, dimension, block, insert, LWPOLYLINE or 3D.
  Text is a single line. Arc span 0 is refused (use a circle).
- Abstention: empty drawings, unsupported entities, foreign SVG profiles, failed
  verification or failed round-trip.
- Invalid input: unknown keys, non-finite numbers, missing layers in the table,
  non-positive radii/paper and duplicate layer names raise ValueError.
- Determinism: export order follows declared layers then entity order. No RNG.
- Measured: docs/DRAFTING_GATE.md and benchmarks/manifests/drafting-roundtrip-v1.json
  on five authored fixtures.
