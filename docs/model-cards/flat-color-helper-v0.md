Model card: bellium/hybrid/flat-color-helper:v0

- Task: fill a selected hole with a median colour when the surrounding field is flat.
- Authority: consultative.
- Order: k-NN labels flat vs textured; only a flat field is painted.
- Limits: not shading, not general inpainting; textured regions abstain.
- Evidence: tests/test_flat_color.py.

