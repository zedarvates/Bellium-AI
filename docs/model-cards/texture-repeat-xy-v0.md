Model card: bellium/knn/texture-repeat-xy:v0

- Task: say whether a game texture repeats in X and/or Y, and report the measured periods.
- Authority: consultative. pixels_changed stays 0.
- Inputs: family (texture, tilemap, sprite) plus an RGB image, or a complete feature object.
- Baseline: normalized autocorrelation on the column and row luma profiles. A flat
  profile declares no period, so a constant axis is never called periodic.
- k-NN memory: models/knn/visual/texture-repeat-v0.json, 25 synthetic fixtures rebuilt
  by scripts/build_game_memories.py. Labels are authored from how each fixture was built.
- Agreement: the family-local verdict must match the measured periods. A periodic verdict
  without any measured period, or a nonperiodic verdict against two strong periods, abstains.
- UV repeat: repeat_counts needs an explicit target_extent_px and only counts axes with a period.
- Limits: synthetic fixtures only, no real game asset evaluation. Periods are pixel counts,
  not UV units. No atlas layout, no mip chain, no seam repair (see texture-tileability).
- Evidence: tests/test_texture_repeat.py.
