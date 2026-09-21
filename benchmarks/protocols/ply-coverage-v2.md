# PLY coverage gate v2 — fixed before measurement

The v1 archive refuses any file above 16 MiB and every ASCII file. A survey of
the local PLY corpus found both refusals on real files, so this gate closes them
and measures what changes. The binary wire format and its golden vectors stay.

## Changes under test

1. ASCII PLY: the header is preserved byte for byte and the body is stored either
   verbatim (`opaque`) or transposed column by column (`columns`). Only canonical
   bodies are accepted: one token per declared field, single-space separators,
   `\n` line endings and a trailing newline. Anything else is refused with an
   explicit error so no separator or spacing is ever silently rewritten.
   A header-only file declaring zero vertices is accepted.
2. Size: `pack`, `inspect_ply`, `encode_ply` and `archive_ply` take an explicit
   `limit`. The default stays 16 MiB; the absolute ceiling becomes 256 MiB and is
   enforced on both encode and decode. The container's default guard is unchanged.

Both are exactness features, not compression features. No predictor changes.

## Measurements

- Coverage before/after over the local PLY corpus, with the refusal reason when
  a file is still outside the bounded subset.
- For every accepted file: complete packet bytes versus a bare zlib reference at
  the same level, byte-exact reconstruction and three timing repetitions.
- Where a canonical `.fovea` file exists for the same source, its on-disk size is
  reported beside the lossless archive size. **`.fovea` is a lossy quantised
  format and the archive is byte-exact; the two are reported together, never as
  interchangeable, and no quality claim is made either way.**
- A legacy `ply-archive/1` packet produced by the previous gate must still decode.

## Decision rules

Ship the two changes only if every accepted file round-trips byte for byte, an
oversized file is refused by default and accepted with an explicit limit, the
absolute ceiling is enforced, and every previously produced packet still decodes.
The size comparison is reported as a measurement, not as a win over `.fovea`.

Remaining out of scope: list properties, faces, extra elements, a `.fovea`
writer, Godot/Zig integration, GPU paths and VR.
