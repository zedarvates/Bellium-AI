# Context patch k-NN: bounded small-fill previews

The previous `inpaint_patch_knn` selected a spatially nearest source pixel. Its
`k_neighbors` argument did not affect synthesis. The new implementation compares
the visible patch context, ranks original intact donors, and averages the RGB
centers of the best `k` matches. The task remains an experimental small-fill
preview. A plausible surrounding patch does not prove a correct reconstruction.

## Behavior and integration

- Original donor patches must be wholly outside the original mask. Damaged
  selected colors never influence matching. Filled context can guide later
  boundary layers, but generated pixels never become donor patches.
- Mask values >128 select pixels. Only their RGB values can change. Image mode,
  alpha and pixels outside the selection are preserved for RGB/RGBA inputs.
- Spatial ties are deterministic. Searching in spatial order stops early only
  after `k` zero-error matches: neither score nor the tie-break can then improve.
- Missing nearby support, excessive mask size, insufficient context agreement or
  exhausted work budget returns the original image and zero repaired pixels.
  `discarded_pixels` records a partially computed preview that was rolled back.
- The example CLI propagates **core** abstention and writes no candidate image
  in that case. Its earlier 5% selection cap remains in force.
- Successfully filled assets receive `needs_review` in `AssetPrepPipeline`;
  cutout confidence cannot authorize the content invented by the fill. Failure
  still propagates `escalate`. No remote model is called.

Core bounds: at most 262,144 canvas pixels, 1,024 selected pixels, odd patch size
3..15, search radius 1..64, and `k` 1..16. Default patch/radius/k are 5/25/3.
The default budget is 200,000 donor comparisons (configurable to at most 2 million).
Default `max_context_error=0.05` is an uncalibrated context-distance threshold.
RGBA distance combines visible RGB squared error and alpha mismatch. Averaging
is in stored RGB code values, not linear light; it can smooth or distort details.

The public example is stricter: at most 16,384 canvas pixels, 5% selected pixels,
patch size 3 and radius 8. These bounded defaults do not promise real-time
processing of arbitrary full-resolution artwork.

```bash
python -m examples.tools.image_tool inpaint --input damaged.png --mask selection.png --output new-preview.png
python -m benchmarks.inpaint_context --output new-inpaint-comparison.json
python run_all_tests.py
python -m unittest discover -s examples/tests -v
```

The output PNG path must be new. CLI exit 0 means a candidate was written, 2 is
abstention without output, and 1 is an input/I/O error. Inspect the preview.

## Frozen comparison

The baseline in `benchmarks/inpaint_v0/` preserves both source files verbatim
from Bellium commit `52e16b05aeec85d52730320b2a5755fd23407533`. Do not optimize
these files. Its random global fallback is prohibited during comparison;
the chosen cases have local sources and do not exercise that branch.

The [machine-readable report](evidence/inpaint-context-v1.json) compares 48
procedural RGB cases, seed 23, at 32×32 pixels with 2×2 or 3×3 defects. Both
mechanisms use patch size 5, radius 8 and k=3 (ignored by the old baseline).
Parameters were fixed before this comparison; there is no network training.
These are designed evaluation fixtures, not a private real-world holdout.

Mean absolute RGB error is normalized to 0..1 over the selected pixels. The table
compares **the same cases completed by both methods**. Abstentions are never
treated as repaired images with zero error.

| Family | Paired cases | Old spatial MAE | Context MAE | Outcome |
|---|---:|---:|---:|---|
| Constant | 8 | 0 | 0 | 8 equal |
| X stripes | 8 | 0.15712 | 0 | 6 improved, 2 equal |
| Y stripes | 8 | 0.17937 | 0 | 7 improved, 1 equal |
| Checkerboard | 8 | 0.39865 | 0 | 8 improved |
| Gradient | 8 | 0.01380 | 0.01714 | **8 worse** |
| Noise | 0 | — | — | Context abstains on all 8 |

The old baseline completes 48/48; context matching completes 40/48. Across the
40 common completed cases, 21 improve, 11 tie and 8 worsen. Source and unselected
pixels remain unchanged on all cases. Separate RGBA regressions verify alpha
preservation and that `k` affects synthesis.

Recorded CPU median/p95 latency: spatial 0.70/1.09 ms; context 2.05/25.14 ms.
This is a quality trade-off with extra computation, not a speedup claim. Timing
includes a complete inference but excludes image decoding. The allocation probe
is a separate single-case Python measurement, not process RAM or GPU VRAM.

CI replays fixture identities, source hashes normalized to LF, output hashes,
per-case decisions, quality values and the visible gradient regression. Timing
and allocation numbers remain the recorded local measurement. No real Asset
Factory sprites, photographic erasure, GPU/homelab performance, calibrated
confidence, model activation or publication quality is established.

## Next improvements

1. Obtain an annotated real sprite/texture set grouped by source asset. The
   current accessible files did not provide a usable labeled sprite set.
2. Treat this 48-case report as fixed development evidence when improving the
   known gradient weakness; reserve new unseen gradient/texture identities for
   the next final evaluation. Consider deterministic interpolation before adding
   a neural correction.
3. Measure seams, unique-detail loss and false acceptance on real examples.
   Preserve abstention and human review even when context distances are low.
4. Compare nano/micro correction or routing only after a real-data baseline
   exists; the separate alpha-QA weights do not predict repair quality.
