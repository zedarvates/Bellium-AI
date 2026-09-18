Model card: bellium/knn/texture-orientation:v0

- Task: name the repeat direction (grain) of a texture and report its angle and period.
- Authority: consultative. pixels_changed stays 0 and no pixel is rewritten.
- Inputs: family (texture, tilemap, sprite) plus an RGB image, or a complete scan object
  with its features.
- Measurement: directions come from the projection method, which keeps phase and
  therefore resolves a striped pattern at any angle. The two image axes are confirmed by
  the shift-difference method with bilinear sampling, the only one that can see a
  symmetric checkerboard. The true repeat direction carries the smallest period; an axis
  is a grain axis when its dip is exact on both axes (a tile) or when it also carries the
  shortest period.
- Direction resolution: a 15 degree coarse scan refined in 2 degree steps, with the
  refinement required to keep the coarse period. Measured worst error on held-out rotated
  cases: 5 degrees.
- Memory: models/knn/visual/texture-orientation-v0.json, 45 authored grains
  (3 families x 5 classes x 3 variants) rebuilt by scripts/build_game_memories.py. The
  builder refuses any fixture whose authored class the measurement contradicts.
- Agreement: the memory must confirm the class the published axis rule computed,
  otherwise the specialist abstains (baseline_disagrees, mixed_neighbor_grains,
  too_few_similar_textures).
- Perspective: when the period changes across the image (ratio above 1.15 in three
  bands), the specialist abstains with repeat_varies_across_image instead of reporting
  one global period.
- Measured on held-out cases: grain correct 15 of 15 decided cases, 0 invented periods,
  perspective refused or reported as no repeat in 3 of 3 warped cases.
  See TEXTURE_REPEAT_GATE.md.
- Limits: pixel-space periods only, no UV or engine units, no anisotropic filtering, no
  mip chain, no claim about how the texture tiles. A mild warp below the variation
  threshold can still be measured as a single period.
- Evidence: tests/test_texture_orientation.py, scripts/benchmark_texture_repeat.py.
