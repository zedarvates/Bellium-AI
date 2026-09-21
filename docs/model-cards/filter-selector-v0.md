# Model card: bellium/knn/filter-selector:v0

- Task and bounded inputs: chooses which convolution preset to apply, from compact
  statistical descriptors of the image rather than from any run of the filters
  themselves.
- Public API and version: `bellium.knn.filter_selector.extract_filter_features(
  image)` and `select_filter_knn(image, *, presets=None) -> SpecialistResult`. v0.
- Authority mode: `consultative`. The specialist names a preset; applying it is
  the caller's decision.
- Model / exemplar source revision and license: authored exemplar set in
  `models/knn/visual/filter-presets-v0.json` (8 presets), MIT, no external
  corpus.
- Training data and seeds: none. The memory is loaded at call time and can be
  replaced through `presets`.
- Independent evaluation data: none specific to this memory. The preset family is
  exercised by `tests/test_magick.py` (`test_convolutions_and_filtering`).
- Deterministic baseline: the caller can apply any preset directly; the selection
  only orders the candidates. The published presets map onto
  `bellium.editing.convolve` (`sharpen`, `gaussian_blur`, `sobel_edges`,
  `emboss`).
- Quality, false agreements and abstention: the specialist abstains with reason
  `no_presets_found` when the memory is empty, rather than defaulting to a filter.
  No accuracy, precision or confusion figure is published for the selector, so a
  wrong preset cannot currently be bounded.
- Hardware and software versions: Windows workstation, CPython 3.14.
- Latency, cold/warm policy, repetitions: not measured.
- RAM / VRAM: not measured.
- Escalation and invalid input behavior: the result carries the authority mode, the
  chosen preset and its distance; nothing is executed.
- Known limitations: only 8 authored presets, no measured selection accuracy, and no
  guarantee that the nearest preset is perceptually the best one.
- Evidence paths and reproduction command:
  `python -m pytest tests/test_magick.py -q`.
