# PLY coverage gate v2 — 18 September 2026

The v1 archive refused two things that real files actually use: ASCII PLY and
anything above 16 MiB. A survey of the local corpus found both. This gate closes
them and records what changed, including where the archive loses.

## What changed

ASCII PLY is now accepted. The header is preserved byte for byte and the body is
stored either verbatim (`opaque`) or transposed column by column (`columns`),
with automatic selection between the two. Only canonical bodies pass: one token
per declared field, single-space separators, `\n` line endings and a trailing
newline. Anything else is refused, so no separator or spacing is ever silently
rewritten. A header-only file declaring zero vertices is valid.

Size became an explicit, bounded declaration. `pack`, `inspect_ply`, `encode_ply`
and `archive_ply` take `limit`; the default stays 16 MiB and the absolute ceiling
is 256 MiB, enforced on both encode and decode. The binary wire format, its
golden vectors and every previously produced packet are untouched: the archived
`ply-archive/1` packet from the previous gate still decodes and re-derives its
original SHA-256.

## Coverage

Eight local files, seven accepted and one refused. Both ASCII files that were
refused before are now accepted, and the 18,709,904-byte capture that exceeded
the old ceiling is accepted with an explicit limit.

| Asset | Source bytes | Accepted | Complete packet | Bare zlib | Encode ms | Decode ms |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| horse runtime (25,674 splats, 59 fields) | 6,060,540 | yes | 5,237,992 | 5,640,439 | 2571.6 | 42.0 |
| reference 3DGS (8,000 splats, 17 fields) | 544,414 | yes | 461,653 | 503,930 | 236.8 | 4.6 |
| demo bonsai (12,473 splats, 14 fields) | 698,849 | yes | 278,451 | 294,345 | 262.5 | 4.3 |
| pathological NaN (64 splats) | 4,764 | yes | 4,323 | 4,184 | 186.7 | 0.2 |
| truncated header | 89 | **no** | — | — | — | — |
| ASCII CLI fixture (1 splat, 14 fields) | 449 | yes | 416 | 189 | 17.0 | 0.1 |
| ASCII header-only (0 vertices) | 49 | yes | 276 | 56 | 1.5 | 0.1 |
| restricted capture (79,273 splats, 59 fields) | 18,709,904 | yes | 14,130,044 | 14,129,835 | 8143.1 | 94.1 |

Every accepted file reconstructed byte for byte, three times each. The refusal is
the negative fixture, which must keep failing.

The layout win is asset-dependent and the envelope costs a fixed few hundred
bytes. The archive is smaller than a bare zlib of the same file on the three
largest splat assets — 8.1%, 8.4% and 5.4% — but **larger on four of the seven**
accepted files: +139 bytes on the NaN fixture, roughly twice the source on the
ASCII CLI fixture, five times the source on the 49-byte header-only file, and
+209 bytes on the restricted capture, where the column layout found no gain at
all. Small files are dominated by the packet header, metadata and two SHA-256
digests. The codec is for assets, not for tiny files.

## Comparison with `.fovea`

Two sources have a canonical `.fovea` file on disk, so the comparison is measured
rather than quoted.

| Asset | Source PLY | Lossless archive | `.fovea` | `.fovea` as a fraction of the archive |
| --- | ---: | ---: | ---: | ---: |
| reference 3DGS | 544,414 | 461,653 | 164,008 | 0.355 |
| demo bonsai | 698,849 | 278,451 | 235,586 | 0.846 |

`.fovea` is a lossy quantised container: it stores palette and covariance
codebooks and drops the higher-order spherical harmonics. The archive is
byte-exact and keeps all 17 or 14 declared fields. The two solve different
problems and are reported together only so the trade is visible. Nothing here
claims the archive competes with `.fovea` on size, and nothing here measures
rendered quality.

## Measurement caveat

Packet sizes depend on the linked zlib build. The previous gate ran on
`1.3.1.zlib-ng`; this run used `1.2.13`, so the same input compressed to 5,640,439
bytes against 5,636,754 before, a 0.065% difference with identical code and
identical source SHA-256. Every report therefore records the zlib identity beside
its numbers, and cross-session comparisons are only made within one build.

## Limits

One local corpus, several assets from the same project, no population sampling.
The restricted capture is measured locally and is not redistributable. ASCII
support covers one vertex element with scalar properties; list properties, faces
and additional elements are still refused. No `.fovea` writer, no Godot or Zig
integration, no GPU path, no VR validation and no rendered-quality measurement.

## Next gate

Extend to the polygonal layout that this gate still refuses: a `vertex` element
plus a `face` element with list properties, which is what most non-splat PLY
files in the wild actually are. Then re-run the same coverage table.
