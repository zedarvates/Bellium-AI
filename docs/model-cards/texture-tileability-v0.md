Model card: bellium/knn/texture-tileability:v0

- Task: rate wrap continuity per axis and report whether a texture meets itself.
- Authority: consultative. pixels_changed stays 0.
- Baseline: per-axis gap between the wrap step and the median local edge step, gated on a
  visible absolute discontinuity (bellium/deterministic/seam-threshold:v0).
- k-NN memory: models/knn/visual/tileability-v0.json, 78 axis records over 39 synthetic
  fixtures rebuilt by scripts/build_game_memories.py.
- Agreement: the deterministic baseline and the family-local neighbors must agree on an axis,
  otherwise that axis abstains. High-frequency noise therefore wraps cleanly instead of
  being reported as broken, and a warm shade is never mistaken for a seam.
- Limits: synthetic fixtures only. A single damaged column next to the wrap plane keeps a
  local reference as large as the seam and abstains; that case needs human review.
- Evidence: tests/test_tileability.py.
