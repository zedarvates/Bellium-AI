# Polygonal PLY gate v3 — scope and decision rules

The v2 archive accepted exactly one vertex element. Most non-splat PLY files are
polygonal: a `vertex` element followed by a `face` element whose property is a
list of indices. Those files were refused. This gate removes that refusal.

Written alongside the implementation; the decision rules below were fixed before
the measurement run and were not changed after seeing its numbers. The
implemented behaviour was already covered by tests when this was written.

## Changes under test

1. Any number of elements, up to 64, each with scalar or list properties, up to
   256 properties per element. List counts are read and bounds-checked with the
   declared count type and byte order, so every element range is known; the item
   values themselves stay opaque.
2. Byte-plane transposition is applied per element instead of to the whole body.
   An element is transposed when its records have a uniform width: always for
   scalar properties, and for list properties only when every record repeats the
   same arity. Mixed arity is archived verbatim inside that element.
3. Each transposed range is recorded in the packet as `[start, end, stride]`, so
   the decoder does not have to interpret transposed bytes to find it. Ranges are
   validated for bounds, alignment, stride and overlap, then the reconstructed
   file is re-parsed against its own header and checked against the source
   SHA-256. Packets written by v1 and v2 carry no range list and keep decoding.

Multi-element ASCII stays refused: no local polygonal ASCII file exists to test
the canonical-row rule against, and guessing a format from nothing is worse than
an explicit refusal.

## Measurements

- A deterministic synthetic mesh corpus, because no licensed real polygonal PLY
  is available offline in this workspace. It is labelled synthetic everywhere and
  is not presented as real-world evidence.
- Three shapes: triangles only, a larger triangle mesh, and a mixed triangle/quad
  mesh that must stay verbatim.
- For every shape, every method and both layouts: complete packet bytes, exact
  byte-for-byte reconstruction and three timing repetitions.
- A diagnostic split between transposing the vertex element only and transposing
  every planar element, to show whether face indices carry their own weight.

## Decision rules

Ship only if every shape round-trips byte for byte under every method, a mixed
arity element is left untouched and still round-trips, the maximum element count
and every malformed range are refused, and packets from the previous gates still
decode. The size numbers are reported as measurements on synthetic geometry.

Out of scope: a `.fovea` writer, Godot or Zig integration, GPU paths, VR, and any
claim about real scanned meshes.
