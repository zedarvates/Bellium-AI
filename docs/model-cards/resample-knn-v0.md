# Model card: bellium/knn/resample-knn:v0

- Task and bounded inputs: subpixel gradient matching for oriented edge
  reconstruction during resize. Features are local luminance gradients; the output
  is a refined pixel placement per target position.
- Public API and version: `bellium.knn.resample_knn.load_patterns()` and
  `knn_resample(image, target_height, target_width, *, patterns=None) ->
  SpecialistResult`. v0.
- Authority mode: `consultative`.
- Model / exemplar source revision and license: authored exemplar set in
  `models/knn/visual/resample-patterns-v0.json` (7 patterns, including
  `flat_uniform`), Apache-2.0, with no external corpus.
- Training data and seeds: none. The memory is read from the JSON file at call time
  and can be replaced by the caller through `patterns`.
- Independent evaluation data: none specific to this memory. The resize family is
  covered by `tests/test_magick.py`; the measured resize numbers in the
  ImageMagick replacement manifest belong to the nano tier.
- Deterministic baseline: `bellium.editing.resample.resample_bilinear`.
- Quality, false agreements and abstention: no held-out accuracy is recorded for
  this memory, and no per-track measurement separates its contribution from the
  bilinear baseline. A flat neighbourhood matches the `flat_uniform` exemplar, so
  the refinement is expected to vanish where there is no gradient.
- Hardware and software versions: Windows workstation, CPython 3.14.
- Latency, cold/warm policy, repetitions: not measured separately.
- RAM / VRAM: not measured.
- Escalation and invalid input behavior: the same target-dimension validation as the
  nano path; the caller may supply its own pattern list.
- Known limitations: an authored 7-pattern memory is not a measured corpus, and no
  comparison against the nano edge model has been recorded.
- Evidence paths and reproduction command:
  `python -m pytest tests/test_magick.py -q`.
