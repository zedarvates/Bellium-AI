# Model card: bellium/knn/patch-inpaint:v0

- Task: fill small RGB holes, consultative only.
- Method: deterministic frontier propagation; exemplar distance uses known pixels
  only. Source patches are fully unmasked in the original image.
- Bounds: area at most 12%, span at most 45% of image size; configurable odd patch,
  local search radius and neighbor count. Inputs are not changed in place.
- Result: filled count, residual mask, complete/partial/uncertain status and abstention.
  Residual holes always abstain. Complete filling does not establish visual truth.
- Local quality check: replay up to four translated holes in known context; at least
  two complete probes, all with MAE <=12/255. The probe budget is 256 missing pixels.
  Insufficient or poor context causes abstention. This is not a statistical guarantee.
- Confidence: None for candidates subjected to these checks. The former heuristic
  remains available as output.support_score; it is not a calibrated probability.
- Evidence: tests for holes of several thicknesses and masked-distance invariance.
- Independent pilot v1 failed (4 severe accepted errors / 24 cases). The v2 guard
  accepted 11/24 new photographic cases with no severe accepted errors, but failed
  the 80% coverage gate. See docs/VISUAL_QUALITY_V2.md. Text and RAM remain unmeasured.
