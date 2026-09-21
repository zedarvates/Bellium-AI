# Polygonal PLY gate v3 — 18 September 2026

The v2 archive accepted exactly one vertex element, so every polygonal PLY — a
`vertex` element followed by a `face` element whose property is a list of indices
— was refused. This gate removes that refusal and keeps the file exact.

## What changed

Any number of elements are parsed, up to 64, each with scalar or list properties,
up to 256 per element. List counts are read with their declared type and byte
order so every element range is known and bounds-checked; the item values stay
opaque. A count that runs past the body, or an absurd count such as four billion
indices, is refused rather than allocated.

Byte-plane transposition is now applied per element instead of to the whole body.
An element is transposed when its records all have the same width: always for
scalar properties, and for list properties only when every record repeats the
same arity. A mesh mixing triangles and quads is archived verbatim inside that
element and still round-trips.

Each transposed range travels in the packet as `[start, end, stride]`. The
decoder does not have to interpret transposed bytes to find them: it validates
bounds, alignment, stride and overlap, un-transposes, then re-parses the result
against its own header and checks the source SHA-256. Packets written by the two
previous gates carry no range list and still decode — verified against the
archived 5,233,969-byte packet from 17 September.

Multi-element ASCII stays refused. No polygonal ASCII file exists offline here to
test the canonical-row rule against, so the codec refuses rather than guesses.

## Measured result (synthetic geometry)

Five deterministic synthetic meshes, every method and both layouts, three
repetitions each, all byte-for-byte exact. **No licensed real polygonal PLY is
available offline in this workspace, so these shapes are generated from formulas
and the absolute ratios are inflated by that regularity. They demonstrate the
mechanism, not real-scan behaviour.**

| Shape | Source bytes | Bare zlib | Automatic packet | Selected |
| --- | ---: | ---: | ---: | --- |
| triangles, 96 vertices | 1,770 | 873 | 510 | delta |
| triangles + normals, 480 vertices | 13,858 | 4,169 | 639 | delta |
| triangles, 4,800 vertices | 78,606 | 34,871 | 917 | delta |
| mixed triangle/quad, 640 vertices | 12,684 | 5,343 | 2,591 | k-NN |
| triangles, big endian, 300 vertices | 5,101 | 2,415 | 561 | delta |

The diagnostic split is the part worth keeping. On the 4,800-vertex mesh,
transposing the vertex element alone gives 12,212 bytes under zlib, while
transposing the face element as well gives 3,784 — the indices carry 69% of that
reduction. On the 480-vertex mesh with normals: 1,998 down to 1,369. Transposing
faces earns its place.

The mixed-arity shape behaves as designed: its face element has stride 0, is not
transposed, and the vertices-only and all-elements diagnostics are identical at
2,625 bytes.

The learned byte predictors stay capped at 64 KiB per packet, so the two largest
shapes record them as skipped rather than silently falling back.

## Limits

Synthetic geometry only. No real scanned mesh, no `.fovea` comparison for
polygonal data, no Godot or Zig integration, no GPU path, no VR validation. The
range list costs roughly 25 bytes per transposed element, which is visible on the
smallest shape. ASCII multi-element files remain unsupported.

## Next gate

Obtain a licensed real polygonal PLY — a scan or an exported mesh with faces —
and re-run this table against it. Then extend ASCII to the polygonal layout using
the same canonical-row rule, once a real ASCII mesh exists to test it against.
