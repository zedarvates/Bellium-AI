Model card: bellium/hybrid/sprite-frame-prep:v0

- Task: build a padded sprite frame from a declared mask, with the anchor placed in it.
- Authority: consultative. The source image is untouched and source_preserved is reported.
- Order: mask validation, content box, clipping refusal, then the k-NN anchor proposal.
- Refusals: a silhouette that already touches the frame returns unsafe /
  content_touches_frame without producing an image. An empty mask abstains. A missing mask
  raises instead of being inferred silently.
- Deterministic checks: the declared content box is copied bit-exactly into the frame and the
  reconstruction is compared before the result is returned. kept_foreground_pixels is reported.
- Anchor fallback: when the k-NN abstains, a declared deterministic default anchor is used and
  the source is reported as deterministic_default with lower confidence.
- Limits: geometry and padding only. No rig, no engine pivot convention, no atlas packing, no
  sprite-sheet slicing, no validated engine import.
- Evidence: tests/test_sprite_prep.py.
