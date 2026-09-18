Model card: bellium/hybrid/texture-tile-fixer:v0

- Task: propose a seam-feathered candidate for a wrap plane that every tier agrees is broken.
- Authority: consultative. The source image is never modified and the candidate is returned
  as a new image with certified false.
- Order: deterministic baseline and family-local k-NN first; the nano model must agree or
  abstain. A nano "continuous" verdict blocks the repair and the axis is left untouched.
- Correction: a mirrored feather across both sides of the wrap plane, weight (band - step) /
  (2 * band), so the plane closes fully and the correction fades inside. band_px is bounded by
  one quarter of the smaller side, default 8.
- Reporting: band_px, changed_channels, max_channel_delta, gap before and after, and improved.
- Limits: geometric continuity only. Perceptual quality of the blend, mip behaviour, normal
  maps and atlas padding are not evaluated. Synthetic fixtures only, no real asset benchmark.
- Evidence: tests/test_texture_tile.py.
