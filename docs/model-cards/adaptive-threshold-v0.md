# Model card: bellium/knn/adaptive-threshold:v0

- Task and bounded inputs: names the binarization strategy for one image, choosing
  between a global Otsu threshold and a local adaptive window from compact image
  descriptors.
- Public API and version:
  `bellium.knn.adaptive_threshold.extract_threshold_features(image)` and
  `knn_threshold(image, *, patterns=None) -> SpecialistResult`. v0.
- Authority mode: `consultative`. The specialist names a strategy; the mask is
  produced by `bellium.editing.morphology`.
- Model / exemplar source revision and license: authored exemplar set in
  `models/knn/visual/threshold-patterns-v0.json` (5 patterns, including
  `clean_text`), Apache-2.0, no external corpus.
- Training data and seeds: none. The memory is loaded at call time and can be replaced
  through `patterns`.
- Independent evaluation data: none specific to this memory. The threshold family is
  exercised by `tests/test_magick.py` (`test_morphology_and_thresholding`).
- Deterministic baseline: `bellium.editing.morphology.otsu_threshold` and
  `adaptive_local_threshold`, both reachable without the memory.
- Quality, false agreements and abstention: an image without usable contrast abstains
  with reason `image_lacks_contrast_for_binarization` and reports the features it
  measured, instead of emitting an arbitrary mask. No accuracy is published for the
  selection itself, so a wrong strategy cannot currently be bounded.
- Hardware and software versions: Windows workstation, CPython 3.14.
- Latency, cold/warm policy, repetitions: not measured.
- RAM / VRAM: not measured.
- Escalation and invalid input behavior: abstention carries the features; a
  non-abstaining result carries the chosen strategy and the authority mode.
- Known limitations: 5 authored exemplars, no measured strategy accuracy, and no
  evaluation on real scanned documents.
- Evidence paths and reproduction command:
  `python -m pytest tests/test_magick.py -q`.
