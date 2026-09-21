# Model card: bellium/hybrid/magick-replacement:v0

## Purpose and Scope

This specialist suite replaces ImageMagick operations with pure-Python, zero-C-dependency,
deterministic baselines augmented with k-NN, nano-NN, and micro-NN specialists following
the Bellium AI hierarchy:

1. Deterministic baseline / exact integer and floating-point math.
2. k-NN exemplar matching in perceptual and spatial feature spaces.
3. Nano-NN / micro-NN inference under strict parameter and byte budgets.
4. Fail-closed abstention on out-of-domain inputs or unconstrained transformations.

## Capabilities and Specialist Mapping

| Operation | ImageMagick Command | Bellium Architecture | Specialist ID |
| :--- | :--- | :--- | :--- |
| **Resize / Resampling** | `convert -resize` | Bilinear baseline + k-NN subpixel patterns + 38-param Nano-NN edge residual | `bellium/nano-nn/resample-edge:v0` & `bellium/knn/resample-knn:v0` |
| **Color Quantization** | `convert -colors N -dither` | Median cut + Floyd-Steinberg + Perceptual k-NN + 25-param Nano-NN dither arbiter | `bellium/nano-nn/quantize-tone:v0` & `bellium/knn/palette-match:v0` |
| **Convolutions & Filters** | `convert -sharpen / -blur / -edge / -emboss` | 3x3/5x5 convolutions + k-NN preset selector + 30-param Nano-NN adaptive blending | `bellium/nano-nn/adaptive-filter:v0` & `bellium/knn/filter-selector:v0` |
| **Tone & Dynamic Range** | `convert -auto-level / -contrast-stretch` | Histogram stretch + equalize + 108-param Micro-NN parametric tone curve | `bellium/micro-nn/tone-curve:v0` |
| **Binarization & Morphology** | `convert -threshold / -morphology` | Otsu threshold + local adaptive + k-NN binarization strategy | `bellium/knn/adaptive-threshold:v0` |
| **Image Comparison** | `compare -metric RMSE/PSNR/SSIM` | Exact MAE, RMSE, PSNR, SSIM and visual diff mask | Deterministic `bellium.editing.compare` |
| **Geometry & Layout** | `convert -rotate / -flip / -flop / -trim` | Exact lossless 90/180/270 rotation, flip, pad, and uniform trim | Deterministic `bellium.editing.geometry` |
| **FastImage Sniffing** | Fast dimension probe | O(1) memory initial bytes parser (PNG, JPEG, GIF, WebP, BMP, PPM, TIFF) | Deterministic `bellium.magick.fastimage` |
| **Perceptual Hashing** | `identify -format "%#"` | 64-bit aHash, dHash, and DCT-based pHash + k-NN nearest exemplar retrieval | `bellium/knn/hash-matcher:v0` |
| **Layer Compositing** | `composite -compose` | 11 blend modes (over, multiply, screen, overlay, dodge, etc.) with opacity and mask | Deterministic `bellium.editing.composite` |
| **Channel Operations** | `convert -separate / -combine` | Extract single channel (R, G, B, Luma) or combine 3 channels | Deterministic `bellium.editing.channels` |
| **Color Spaces** | `convert -colorspace` | Exact conversions sRGB â†” HSV and sRGB â†” CIE L*a*b* + Delta-E CIE76 | Deterministic `bellium.editing.colorspace` |
| **Montage & Contact Sheet**| `montage` | Contact sheets, sprite grids and collages with tile sizing and padding | Deterministic `bellium.editing.montage` |
| **EXIF Auto-Orientation** | `convert -auto-orient` | Automatic lossless orthogonal transform based on EXIF tag (1-8) | Deterministic `bellium.magick.auto_orient` |
| **Raster to SVG** | `convert file.svg` / autotrace | Quantize, pixel-square contours, k-NN region routing; photos abstain | `bellium/hybrid/raster-to-svg:v0` |
| **Geometry Syntax** | `800x600^`, `50%`, `800x600#` | ImageMagick geometry specification parser | Deterministic `bellium.magick.geometry_parser` |

## Model Budgets and Governance

- `bellium/nano-nn/resample-edge:v0`: 38 parameters (budget: 64), 6 inputs, 2 outputs. Predicts subpixel edge sharpness residual.
- `bellium/nano-nn/quantize-tone:v0`: 25 parameters (budget: 48), 4 inputs, 1 output. Controls dithering strength to prevent banding on subtle gradients while suppressing noise in flat areas.
- `bellium/nano-nn/adaptive-filter:v0`: 30 parameters (budget: 48), 4 inputs, 2 outputs. Blends between sharpening structural edges and smoothing noise.
- `bellium/micro-nn/tone-curve:v0`: 108 parameters, 8 global distribution inputs, 4 parametric curve outputs.
- All nano models are verified via `inspect_model()` against `NanoBudget`.
- Input images are never mutated in place; pure functional returns.
- Fully test-backed via `tests/test_magick.py`.
