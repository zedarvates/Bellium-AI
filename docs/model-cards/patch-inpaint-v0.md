# Model card: bellium/knn/patch-inpaint:v0

- Task: fill small RGB holes, consultative only.
- Method: deterministic frontier propagation; exemplar distance uses known pixels
  only. Source patches are fully unmasked in the original image.
- Bounds: area at most 12%, span at most 45% of image size; configurable odd patch,
  local search radius and neighbor count. Inputs are not changed in place.
- Result: filled count, residual mask, complete/partial/uncertain status and abstention.
  Residual holes always abstain. Complete filling does not establish visual truth.
- Source search is bounded to the hole box expanded by the search radius, which is
  equivalent to an image-wide scan because no masked pixel looks further. Source
  patches are flattened once per call instead of once per candidate.
- Local quality check: one masked block per probe on the candidate fill, at the four
  nearest displacements whose known ring matches the hole's ring within 18/255. Every
  probe must stay within 12/255 and at least two must complete. The probe budget is
  256 missing pixels and 120 candidate sites. Unmatched context causes abstention.
- Confidence: None for candidates subjected to these checks. The former heuristic
  remains available as output.support_score; it is not a calibrated probability.
- Evidence: tests for holes of several thicknesses and masked-distance invariance.
- Validated on a frozen lot of three never-used photographs (18 cases): coverage
  83.3%, no accepted error above 25/255, mean 5.23/255, 15.6% better than the
  nearest-known-pixel baseline, p95 0.33 s. All six protocol gates pass. Synthetic
  calibration also passes (81.8% of reconstructible cases, 0% noise acceptance).
  Scope is three photographs at 128 px; text, faces, large images and RAM/VRAM are
  not measured. See docs/VISUAL_QUALITY_V4.md.
