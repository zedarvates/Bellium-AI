Model card: bellium/knn/clip-loop:v0

- Task: say whether a sprite or effect clip closes its loop, from the first and last frame.
- Authority: consultative. pixels_changed stays 0 and no clip is retimed or exported.
- Inputs: family (character, effect, ui, prop) plus a first/last frame pair with image and
  mask, or the complete feature object.
- Features: iou_mismatch, added_ratio, removed_ratio, centroid_shift, area_ratio,
  color_delta, bbox_shift. All in [0, 1]; missing or out-of-range values raise.
- Baseline: bellium/deterministic/clip-loop-threshold:v0 (mask mismatch, centroid and
  bounding-box shift thresholds). The baseline must agree with the k-NN verdict.
- Reference set: models/knn/layout/clip-loop-v0.json, 24 authored pairs rebuilt by
  scripts/build_game_memories.py. Only items labelled loops form the reference space.
- Decision: inside the known loop space means loops; clearly outside it with a drifting
  baseline means drifts; the band between the two abstains with between_known_spaces.
- Limits: authored synthetic pairs, no artist-reviewed clips and no engine timeline.
  A small drift next to the loop space abstains by design; it needs human review.
- Evidence: tests/test_clip_loop.py.
