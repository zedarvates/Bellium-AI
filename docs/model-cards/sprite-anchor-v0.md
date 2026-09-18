Model card: bellium/knn/sprite-anchor:v0

- Task: propose an anchor kind and position for a sprite inside its content box.
- Authority: consultative. pixels_changed stays 0 and no frame is cut.
- Inputs: family (character, effect, ui, prop) plus an RGB image with a foreground mask, or a
  complete feature object.
- Features: coverage, bbox_fill, centroid_x, centroid_y, bottom_mass, symmetry, aspect.
- k-NN memory: models/knn/visual/sprite-anchor-v0.json, 16 authored silhouettes in four
  families, rebuilt by scripts/build_game_memories.py.
- Agreement: at least three similar neighbours of one family must agree on the anchor kind;
  mixed anchors abstain. The position is the mean of the agreeing neighbours and the spread
  is reported.
- Limits: authored silhouettes only, no artist-reviewed rigs. Anchors are content-box ratios,
  not engine pivots, and they are never certified. Character, prop, effect and UI are the only
  supported families.
- Evidence: tests/test_sprite_prep.py.
