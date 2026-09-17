# Bounded gradient interpolation before context k-NN

`inpaint_preview` adds a deterministic bilinear RGB candidate for small defects
with regular surrounding color. The example image CLI and `AssetPrepPipeline`
now use it. `inpaint_patch_knn` remains available unchanged, and its earlier
[comparison and eight gradient regressions](INPAINT_CONTEXT_QA.md) remain frozen.

## Contract

Four original, unselected corners of an enclosing rectangle define the RGB
interpolant. Every other known pixel inside that rectangle validates it: the
maximum residual must be **at most one stored RGB code value** in every channel.
The selected RGB values never affect estimation or validation.

- The selection bounding box must fit within 16x16 pixels, with two pixels of
  real context on all sides. The examined rectangle is at most 20x20. Sparse
  selections are allowed but do not expand this bound.
- RGB and RGBA inputs are supported. All alpha inside that rectangle, including
  selected alpha, must be equal and positive. Alpha itself is never repaired.
- Integer bilinear weights and nearest-integer rounding, with half upwards,
  avoid floating-point ties. Computation uses stored RGB, not linear light.
- A rejected interpolation gate calls the unchanged context k-NN with the same
  parameters. Its budget, rollback, alpha preservation and abstention remain.
- General core limits remain 262,144 canvas pixels, 1,024 selected pixels and
  a 25% selection ratio. k-NN keyword arguments are validated even when
  interpolation succeeds; the comparison budget controls the fallback only.
- The CLI keeps its stricter 16,384-pixel canvas / 5% selection limit and
  patch=3, radius=8, k=3. It writes a new PNG only when the core returns a candidate.

`metrics.method` identifies `bilinear_rgb_v1` or `context_patch_knn_v1`.
`metrics.interpolation` records acceptance, rejection reason, support and
validation counts, and maximum residual when computed. Successful interpolation
uses route method `bilinear_rgb` and confidence 0: this gate has no calibrated
probability of reconstruction correctness. `PREVIEW_METHODS` names the two local
candidate routes. Original mode, alpha and unselected pixels are preserved.

The pipeline retains **needs_review** for successful repair unless another gate
requires escalation. Core abstention propagates escalation; the CLI writes
nothing. A low residual cannot establish the presence or absence of hidden
content. No remote model or nano/micro classifier participates in this fill.

```python
from bellium.inpaint import inpaint_preview

result = inpaint_preview(image, defect_mask)
print(result.metrics.method, result.metrics.interpolation)
```

```bash
python -m examples.tools.image_tool inpaint --input damaged.png --mask selection.png --output new-preview.png
python -m benchmarks.inpaint_preview --output new-preview-comparison.json
python run_all_tests.py
python -m unittest discover -s examples/tests -v
```

## Fixed evaluation

The [protocol](../benchmarks/preview_protocol.md) and mechanism were committed
locally before implementing and running the new comparison. The evaluator checks
the pre-evaluation mechanism SHA-256
`89f32baa31ed60d4576ade795feb0016886c17f57af97438350ddc9342ec7044`.
No algorithm, threshold or fixture adjustment followed the first result.

The [recorded report](evidence/inpaint-preview-v1.json) has two distinct groups:

1. **48 development cases** reused from the earlier comparison. The eight
   gradients improve from mean masked MAE 0.01714 to **0**. The other 32 completed
   cases are identical to context k-NN; all eight noise cases still abstain.
2. **96 new identities**, seed 104729, twelve families with eight variants each.
   Images are 36..44 pixels per side, RGB/RGBA, with 1..4-pixel defect spans and
   rectangular or diagonal selections. Reference hashes are unique and disjoint
   from the development cases. These are procedural evaluation cases, not an
   independent, human-annotated Asset Factory dataset.

Both mechanisms use patch=5, radius=8, k=3, max_comparisons=200,000 and
max_context_error=0.05 in the comparison. This differs from the public CLI's
patch size. The table uses normalized RGB MAE on **common completed cases**.

| New family | Common completed | Context MAE | Preview MAE | Interpolated | Outcome |
|---|---:|---:|---:|---:|---|
| Integer affine gradients | 8 | 0.00911 | 0 | 8 | 8 improved |
| Quantized affine gradients | 8 | 0.00615 | 0.00012 | 8 | 8 improved |
| Bilinear color fields | 8 | 0.00487 | 0.00059 | 8 | 8 improved |
| Constant | 8 | 0 | 0 | 8 | 8 equal |
| Stripes | 8 | 0 | 0 | 0 | 8 equal |
| Checkerboard | 8 | 0 | 0 | 0 | 8 equal |
| Curved gradients | 8 | 0.09988 | 0.09988 | 0 | 8 equal, still inaccurate |
| Steps | 8 | 0 | 0 | 0 | 8 equal |
| Noise | 0 | — | — | 0 | Both abstain on all 8 |
| Fully hidden detail | 8 | 0.39597 | 0.39632 | 8 | **1 improved, 3 equal, 4 worse; all lose detail** |
| Border gradients | 8 | 0.01244 | 0.01244 | 0 | 8 equal |
| Alpha discontinuities | 8 | 0.01149 | 0.01149 | 0 | 8 equal |

Both complete 88/96 and abstain on eight noise cases. On the common 88, **25
improve, 59 tie and 4 regress**. All 24 new regular-gradient/field cases improve;
their maximum per-channel error is at most one code value. Forty candidates use
interpolation. Fifteen of those are nonexact: seven have only one-code-value
quantization differences, and eight lose completely hidden detail with much
larger errors. The latter are adverse candidates despite matching context.
No completed candidate is automatically accepted into production.

All source, mask, unselected pixels and alpha checks pass. On the 56 new cases
where interpolation declines, outputs, decisions, fill counts, donor comparison
counts and discarded counts exactly match direct context k-NN, including its
eight abstentions. The curved-gradient errors also remain visible.

Recorded local CPU median/p95 is 5.84/38.24 ms for context and 1.92/32.23 ms for
the preview. These are one inference per synthetic case, including gate work and
fallback but excluding decoding. They do not establish general hardware speed.
The separate first-case Python allocation probe is 180,714 versus 4,752 bytes;
it is not full process RAM or GPU VRAM. CI replays inputs, hashes, decisions and
quality; it does not validate these local timing/allocation measurements.

New dataset SHA-256:
`42a0acdf9f9bd05928f3741bb65af8510b150223ca1b6a49f00211eb483a0bc1`.

## Remaining work

Collect real annotated sprite/texture references and defect masks before claiming
Asset Factory quality or selecting a nano/micro repair model. Prioritize curved
shading, antialiased boundaries, unique detail and human rejection rates. The
separate nano/micro alpha-QA classifiers are unchanged and cannot judge whether
RGB repair preserved the intended content. These 96 identities must become
development evidence if future changes are selected using their outcomes.
