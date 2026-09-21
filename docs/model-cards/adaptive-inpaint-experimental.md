# Model card: bellium/hybrid/adaptive-inpaint:experimental-v1

- Task: propose a small-hole fill by choosing between two deterministic methods.
- Authority: consultative. Not registered, not used by the Pillow adapter, never applies itself.
- Methods: patch k-NN copy and interpolation between the nearest known pixels on
  each axis. The hole is never read while scoring.
- Selection: each method must pass every translated-hole probe with MAE <= 12/255,
  on a single masked block over that method's own candidate fill. Probe sites are
  the 120 nearest displacements whose known ring matches the hole's ring within
  18/255. At least two probes per method are required.
- Agreement: both methods must survive. A lone survivor is refused because a single
  lucky neighbourhood can hide an unseen feature such as a point highlight.
- Photographic evidence (24 fresh cases, four CC0/public-domain images): 15/24
  accepted, coverage 62.5%, no accepted case above 25/255, mean masked error 4.23/255,
  25.5% better than the nearest-known-pixel baseline, p95 0.74 s.
- Synthetic evidence: the shared guard accepts 81.8% of reconstructible cases and
  0% of white and blurred noise, with every accepted error under 12/255.
- Limits: the 80% coverage target is not met. A 3x3 hole can hide a bright point
  source that no probe on visible pixels can reveal. Text quality, RAM/VRAM and
  target-device latency are not measured.
- Evidence: tests/test_adaptive_inpaint.py, tests/test_inpaint_guard_calibration.py,
  docs/VISUAL_QUALITY_V3.md.
