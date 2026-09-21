# Model card: white background APIs

- Native API: `normalize_white_background`, consultative RGB cutout and crop.
- Pillow compatibility API: `normalize_background`, install the `image` extra.
- Existing RGBA transparency is preserved. Padding controls the fitted foreground
  size. Unsafe Pillow extraction raises `UnsafeCutoutError` with its metrics.
- Data: current image only. No image is written to disk by these functions.
- Evidence: RGBA, padding, feathering and refusal regression tests.
- Artistic validity, complex scene quality and RAM/VRAM: not measured.
