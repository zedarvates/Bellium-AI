# P4 rasterization gate: exact pixels, verified pages, real PNG bytes

The plan said where the frames go and the manifest said what an importer would read. This
gate writes the pixels, proves them against the plan, and encodes pages an engine could open.
Everything is generated geometry and pure Python, so the whole chain is reproducible offline.

    python scripts/benchmark_atlas_export.py

Report: benchmarks/manifests/atlas-export-v1.json.

## What exists

| Piece | Behaviour |
| --- | --- |
| rasterize | exact copies into a background-filled page; RGB or RGBA, one channel count per atlas |
| check_raster | re-derives copies, background, gutter band, page sizes, digests and placement bookkeeping |
| encode_png | 8-bit RGB or RGBA, filter 0, one IDAT, fixed zlib level: the same page gives the same bytes |
| inspect_png | re-parses the container: signature, chunk CRCs, IHDR fields, IDAT order, IEND position, inflate length, filter bytes |
| decode_png | decodes that same subset back to pixels for a digest round-trip |
| rasterize_atlas | one call: pack, check the plan, check the sources, rasterize, verify, optionally encode |

A page that fails its own check is never returned: the specialist abstains with
raster_failed_verification or png_failed_verification instead.

## Invariants, recomputed

| Invariant | Result |
| --- | --- |
| Every placement holds the source pixels byte for byte | holds |
| Every pixel outside a placement is the declared background | holds |
| The gutter band stays background when padding is declared | holds |
| Page size, page count and raw byte count match the plan | holds |
| Digests describe the pixels they are attached to | holds |
| Encoded pages decode back to the same digest | holds |
| Pillow opens every page and agrees pixel for pixel | holds on all 5 cases |
| Violations across the benchmark | 0 |

## Measured

| Case | Frames | Channels | Pages | Pixels | Raw bytes | PNG bytes | Ratio | Raster | Encode | Verify |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| uniform_16_rgba | 64 | 4 | 1 | 16384 | 65536 | 54406 | 0.830 | 20.7 ms | 4.3 ms | 7.3 ms |
| mixed_rgb | 12 | 3 | 1 | 16384 | 49152 | 19876 | 0.404 | 10.9 ms | 4.5 ms | 6.3 ms |
| mixed_padded_rgba | 12 | 4 | 1 | 36864 | 147456 | 29718 | 0.202 | 14.1 ms | 8.9 ms | 15.7 ms |
| measured_96_rgba | 14 | 4 | 1 | 9216 | 36864 | 24561 | 0.666 | 12.9 ms | 3.5 ms | 5.4 ms |
| two_pages_rgb | 24 | 3 | 2 | 32768 | 98304 | 55806 | 0.568 | 28.9 ms | 9.7 ms | 13.0 ms |

Totals: 397312 raw bytes to 184367 PNG bytes, ratio 0.464, with 0 invariant violations and 0
Pillow mismatches. The ratio is content-dependent by a factor of four across these cases: pages
full of dense gradients compress poorly (0.830) while pages with transparent gaps compress well
(0.202).

## The honest reading

What this gate proves is exactness. Every byte inside a placement is the source pixel, every
other byte is the declared background, and the encoded file decodes back to the same digest —
checked by this repository and independently by Pillow, which opened all five pages and agreed
on every pixel.

What it does not prove is size or speed. The encoder uses one filter type and a fixed zlib
level, so it will lose to a tuned encoder that selects filters per scanline; the measured ratios
show the spread but no comparison against a reference encoder was run. Rasterization and
verification are pure Python and cost 10 to 29 ms for 9k to 37k pixels, which is fine for asset
preparation and far from interactive for large pages.

Not established: engine import of these files, platform texture formats and compression
(mipmaps, ASTC, ETC2, PVRTC), interlacing, 16-bit or palette output, deduplication of identical
frames, atlas page count heuristics for real projects, and any evaluation on real sprite sheets.
The engine round trip stays open in P4: this gate removes "no pixels exist" as an excuse, not
the engine itself.

