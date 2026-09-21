# Real polygonal corpus gate v4 — scope and decision rules

The v3 gate proved element-aware transposition on synthetic meshes and recorded
that no real polygonal PLY was available offline. This gate fixes that: it
measures an external, hash-pinned corpus of third-party meshes, and it closes the
refusal those files exposed.

Written before the measurement run. Decision rules below were not changed after
seeing its numbers.

## Corpus

Eight meshes from the `mikedh/trimesh` test corpus, pinned to one commit, MIT
license verified from that commit. The files stay in a local evidence folder and
are never added to the repository or the wheel. `bunny.ply` and `suzanne.ply` are
excluded because their origins carry separate terms.

Five are binary little-endian polygonal meshes; three are ASCII polygonal meshes.
The ASCII ones matter twice over: they exercise the text path, and two of the
three are **not** canonically spaced, which the previous gate refused outright.

## Changes under test

1. Multi-element ASCII. The header is preserved byte for byte and the body is
   read line by line against each element's declared fields, including list
   arities. Irregular spacing, tabs, CRLF endings or a missing final newline are
   accepted and stored **verbatim**; nothing is reformatted.
2. The column layout is offered only when the body is canonical — every line is
   exactly its tokens joined by one space, with a trailing newline — and every
   element has a uniform row width. Otherwise only `opaque` is available, and an
   explicit `columns` request fails instead of silently falling back.
3. The metadata carries a per-element `shape` of `[count, width]`; the earlier
   single-element `count`/`columns` shape still decodes.

An ASCII record's list arity must be a decimal integer and must match exactly the
number of tokens present, so a truncated or padded row is refused rather than
read loosely.

## Measurements

Every file: structural report, complete packet bytes for each layout and method,
byte-for-byte reconstruction, three timing repetitions, a bare zlib reference and
the automatic selection. Plus the real splat corpus and the archived packets from
the earlier gates, re-decoded to confirm nothing regressed.

## Decision rules

Ship only if every corpus file reconstructs byte for byte, the three ASCII meshes
are accepted, a non-canonical body is never transposed, a tampered shape is
refused, the previous packets still decode, and the full test suite and package
smoke pass. Size numbers are reported as measurements on third-party test meshes;
no claim is made about arbitrary production meshes.

Out of scope: a `.fovea` writer, Godot or Zig integration, GPU paths, VR, and the
ultimate provenance of each third-party mesh beyond its repository licence.
