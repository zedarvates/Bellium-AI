Model card: bellium/hybrid/direct-filters:v0

- Task: replay a declared list of image operations on a source image and return the
  result, with the recipe that produced it.
- Authority: consultative. source_preserved is always true: the stack keeps its own copy
  and never writes to the caller's image.
- Operations: grayscale, sepia, invert, brightness_contrast, saturation, tint and
  threshold, each with documented parameters and an adjustable strength that blends
  between the source pixel and the filtered one.
- Masks: rectangle, ellipse, polygon and brush strokes, with optional feathering; a
  masked step leaves every uncovered pixel byte for byte.
- Reproducibility: the stack exports a recipe and replays it to identical pixels, and
  undo then redo returns the same image.
- Measured: a 64 x 64 preview costs 4 to 18 ms per operation and a 192 x 192 full edit
  34 to 163 ms in pure Python, so the tool is not interactive. See the gate record.
- Backends: a vectorised path when NumPy is importable, 256-entry lookup tables for the
  per-channel filters otherwise, and the plain per-pixel path as the reference. All three
  agree pixel for pixel, which a parametrised test enforces. NumPy is never required.
- Previews: area-averaged to a declared side, reported with scale, size and preview_only,
  and never confused with the export. A preview session converts the source once and then
  repeats previews at 14.7 ms p95 on a 512 x 512 source, inside the declared 33 ms
  budget, while a cold one-shot preview costs about 0.5 s. See the interactive record.
- Limits: no colour management, no curves, no blend modes, no layers, and colours are
  clipped in the source space. No neural tier is used, because every operation is an
  exact formula.
- Evidence: tests/test_editing.py, scripts/benchmark_editing.py.
