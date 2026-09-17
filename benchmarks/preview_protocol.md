# Bilinear preview evaluation protocol

This protocol and `bellium/inpaint/preview.py` are fixed before the first run of
the new fixture comparison. The implementation was checked only against hand
constructed unit/integration cases. The previous 48-case context comparison is
development evidence; it must not be overwritten or renamed as a holdout.

## Fixed mechanism

- Four original, unselected enclosing corners define bilinear RGB interpolation.
- Selection bbox is at most 16x16, with a two-pixel margin on every side. All
  original alpha in that rectangle must be equal and positive. Maximum rectangle
  is 20x20. General 262,144 canvas / 1,024 selection / 25% ratio limits remain.
- All other unselected pixels validate the interpolant with an inclusive maximum
  residual of one RGB code value. There is no fitted/tuned threshold or NN.
- Integer weights and nearest-integer rounding (half upwards), stored RGB space.
- Failed interpolation checks call unchanged `context_patch_knn_v1` with the
  supplied k-NN budgets. Failure there still returns the original image.
- Comparison parameters: patch=5, radius=8, k=3, comparisons=200,000,
  max_context_error=0.05 for both the direct baseline and hybrid fallback.
- The public CLI separately keeps patch=3, radius=8, its 16,384-pixel canvas cap
  and its 5% mask cap. Benchmark latency is not a CLI timing measurement.

## New evaluation identities

Use seed 104729, eight variants in each of twelve families, 36..44-pixel-wide
and high RGB/RGBA images and 1..4-pixel defect spans. Families: integer affine,
quantized affine, bilinear, constant, stripes, checkerboard, curved gradients,
steps, noise, fully hidden detail, border gradients and alpha discontinuities.
Include nonrectangular selections. Publish reference/mask/damaged hashes and
check exact reference hashes are disjoint from the 48 old development cases.

Evaluate context v1 and the hybrid once per case, recording actual method,
completed/abstained counts, gate diagnostics, source/mask/outside/alpha
preservation, output hashes, masked and boundary RGB MAE. Compute improvement
only on common completed pairs and expose any coverage change. Score abstentions
as null, never as zero reconstruction error. Record CPU time including gates;
the separate one-case Python allocation probe is not process RAM or GPU memory.

Report harmful interpolation candidates (completed interpolation with nonzero
masked error), maximum RGB error and fully hidden detail losses, including ties
or regressions against k-NN. A fully hidden mark may share exactly the same
observed context as an undamaged gradient; no local gate can establish its
absence. `needs_review` therefore remains mandatory even for zero residual.

Do not tune the interpolation, thresholds, fallback or fixtures after inspecting
this result. If a correctness bug requires a change, record it and treat these
identities as development cases for the next version. This is a procedural
evaluation with newly generated identities, not independent human-labeled
Asset Factory evidence. Nano/micro alpha classifiers stay separate.
