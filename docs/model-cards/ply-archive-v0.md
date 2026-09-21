# Model card: bellium/hybrid/ply-archive:v0

- Task: archive a complete supported binary PLY file so every original byte is
  recovered, including header text, field order, spherical-harmonics
  coefficients and unknown scalar attributes.
- Tier: deterministic container, consultative only, `decision_eligible` false.
- Method: reversible byte-plane transposition of the vertex records, then BLCP/1.
  Interleaved records are compared against the transposed layout and the smaller
  complete packet is kept. No quantisation, no value conversion, no field dropped.
- Evidence: `tests/test_ply_archive.py`, `scripts/benchmark_ply_archive.py`,
  the external evidence run `compression-full-ply-2026-09-17`.

## Supported subset

Binary little- or big-endian **and ASCII** PLY 1.0 with up to 64 elements, each
with scalar or list properties. All declared widths are read:
char/uchar/short/ushort/int/uint/float/double and their int8..float64 aliases.
Limits are 256 properties per element, a 16 KiB header, a
caller-declared file limit and 9 digits of element count. The default file limit
is 16 MiB; `limit` raises it to an absolute ceiling of 256 MiB, and the same bound
must be passed on decode.

Polygonal files are supported: a `face` element with a list of indices is read
and archived exactly, and transposed as well when every face repeats the same
arity. List counts are bounds-checked, including absurd counts. Refusal is the
intended behaviour for anything outside the subset; the codec never guesses.
Every ASCII body is stored verbatim (`opaque`), so irregular spacing, tabs, CRLF
endings or a missing final newline survive untouched and are never reformatted.
The `columns` layout is offered only when the body is canonical — each line is
exactly its tokens joined by one space, with a trailing newline — and every
element has a uniform row width; an explicit `columns` request on anything else
fails instead of falling back silently. A record whose list arity does not match
the tokens present is refused, even when the file came from a real exporter.

Values are treated as opaque bytes. A NaN payload, a signalling float, an
unexpected quaternion or a custom vendor column all survive unchanged. The
separate `encode_splats`/`decode_splats` API remains the option when the caller
wants numeric validation and explicit-base temporal residuals.

Packet sizes depend on the linked zlib build, so every benchmark report records
the zlib identity beside its numbers. See
[the coverage gate](../PLY_COVERAGE_V2.md) for the measured table, the `.fovea`
trade-off, the files where the archive is larger than a bare zlib, and the
ASCII/size changes, then [the polygonal gate](../PLY_POLYGONAL_V3.md) for
element-aware transposition and its synthetic-only measurement.

## Measured result

Run of 17 September 2026 on the complete Horse Statue derived reconstruction,
SHA256 `4392920074f41204ec900244da06505a8696312ea989340df88b1bd9c025806a`,
6,060,540 bytes, 1,476-byte header, 25,674 vertices, 236 bytes per record,
59 float properties. Three timing repetitions per case; every case reconstructed
the whole file byte-for-byte.

| Case | Packet bytes | Encode ms | Decode ms |
| --- | ---: | ---: | ---: |
| Bare zlib reference | 5,636,754 | 132.2 | 15.7 |
| Original records + zlib | 5,636,963 | 145.9 | 30.1 |
| Original records + delta | 5,984,189 | 800.7 | 707.3 |
| Byte planes + zlib | **5,233,969** | 142.3 | 36.8 |
| Byte planes + delta | 5,320,643 | 787.4 | 703.3 |
| Archive, automatic | 5,233,969 | 1848.1 | 35.3 |

Byte planes with zlib is 7.15% smaller than the bare zlib reference and 13.64%
smaller than the source file. Automatic selection reached the same packet size
but paid to encode every candidate; a caller that knows the file size profile
should force the layout and method. Delta residuals lose on both layouts here.

`tracemalloc` peaks for the forced winner were 21,769,035 bytes to encode and
24,247,092 bytes to decode. That measurement excludes the caller-owned input and
is not process RSS, native library memory, VRAM or a target-device budget.

## Limits

This is one known integration asset whose training source is synthetic multiview
footage. It is not a new held-out sample and not a natural motion sequence. No
comparison against the existing `.fovea` codec, no Godot or Zig integration, no
GPU path and no VR validation has been performed. Learned predictors stay capped
at 64 KiB per packet, so this large-file result says nothing about a neural win.

## Next gate

Run the same archive over several independently licensed real-world scans with
different property counts, then compare total bytes and cost against the existing
`.fovea` codec and a native implementation. Until that exists, this codec is a
consultative archival option, not a production format decision.
