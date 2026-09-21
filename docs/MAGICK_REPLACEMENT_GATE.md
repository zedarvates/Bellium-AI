# Gate: ImageMagick Replacement v1

## Objective

Replace ImageMagick C binaries with native, zero-dependency Bellium AI specialists obeying
the strict hierarchy: deterministic baseline â†’ k-NN exemplar matching â†’ nano-NN / micro-NN inference.

## Measured Protocol Evidence (`benchmarks/manifests/magick-replacement-v1.json`)

| Track | Specialist ID | Method | Latency (p50) | Verification Property |
| :--- | :--- | :--- | :--- | :--- |
| **Resize (Edge)** | `bellium/nano-nn/resample-edge:v0` | 38-param Nano-NN | ~10-15 ms | In smooth areas (`gradient`), `refined_pixels = 0`, `diff_mae = 0.0`, `diff_ssim = 1.0` (exact bilinear convergence). On `lineart`, refines 2,811 pixels along structural edges. |
| **Quantize (Tone)** | `bellium/nano-nn/quantize-tone:v0` | 25-param Nano-NN | ~4-6 ms | Modulates dither intensity dynamically (0.78 on smooth gradients, 0.99 on sharp lineart) preventing banding while preserving clean edges. |
| **Adaptive Filter** | `bellium/nano-nn/adaptive-filter:v0` | 30-param Nano-NN | ~25 ms | Blends sharp vs smooth weights: 93.8% sharpening on lineart, 68.3% smoothing on flat/noisy zones. |
| **Tone Curve** | `bellium/micro-nn/tone-curve:v0` | 108-param Micro-NN | ~1.8 ms | Parameterizes continuous gamma, lift, gain, and pivot without blown-out highlights. |
| **CLI & I/O** | `bellium.magick.cli` | Pure Python + Netpbm/JSON | < 5 ms overhead | Standalone CLI supporting `resize`, `quantize`, `filter`, `tone`, `threshold`, `morphology`, `compare`, `transform`, and `info`. |

## Gate Verdict

- All 11 unit and CLI tests in `tests/test_magick.py` and `tests/test_magick_cli.py` pass.
- Zero native C library dependency; zero network or external tool execution.
- Ruff linter: 0 warnings, 0 errors.
- Status: **PASSED (Consultative)**.
