Model card: bellium/hybrid/atlas-raster:v0

- Task: rasterize a verified atlas plan into pages and hand back either exact pixels or exact
  8-bit RGB/RGBA PNG bytes.
- Authority: consultative. written_files stays false: nothing is written to disk and no engine is
  invoked. certified stays false.
- Method: exact copies. Every pixel inside a placement is the source pixel at the same offset and
  every other pixel is the declared background. Nothing is scaled, cropped, resampled, blended or
  rotated, and alpha travels as a fourth channel.
- Sources: exactly the declared frames, one image each. A missing or extra source, a size
  mismatch, mixed channel counts or malformed pixels abstain with source_contract_violated.
- Verification: check_raster re-derives the pages from the plan and the sources (copies,
  background, gutter band, page sizes, digests, placement bookkeeping). inspect_png re-parses the
  container (signature, chunk CRCs, IHDR fields, consecutive IDAT, IEND position, inflate length
  and filter bytes). Encoded pages are decoded back and compared by digest before they are
  returned.
- PNG subset: 8-bit RGB (colour type 2) or RGBA (6), filter 0, no interlacing, one IDAT and a
  fixed zlib level, so the same page always encodes to the same bytes.
- Measured: benchmarks/manifests/atlas-export-v1.json, 5 cases: 397312 raw bytes to 184367 PNG
  bytes (0.464), 0 invariant violations, 0 Pillow mismatches. Raster 10.9 to 28.9 ms, encode 3.5
  to 9.7 ms and verification 5.4 to 15.7 ms per case.
- Limits: no per-scanline filter selection, no palette, no 16-bit, no interlacing, no mipmaps, no
  platform texture formats, no file writing and no engine execution. Pillow appears in the
  benchmark only.
- Evidence: tests/test_atlas_raster.py, scripts/benchmark_atlas_export.py.

