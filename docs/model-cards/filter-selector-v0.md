# Model card: bellium/knn/filter-selector:v0

- Task and bounded inputs: chooses which convolution preset to apply, from compact
  statistical descriptors of the image rather than from any run of the filters
  themselves.
- Public API and version: `bellium.knn.filter_selector.extract_filter_features(
  image)` and `select_filter_knn(image, *, presets=None) -> SpecialistResult`. v0.
- Authority mode: `consultative`. The specialist names a preset; applying it is
  the caller's decision.
- Model / exemplar source revision and license: authored exemplar set in
  `models/knn/visual/filter-presets-v0.json` (8 presets), Apache-2.0, no external
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

*** Add File: C:\BelliumAI\Bellium-AI\docs\model-cards\tone-curve-v0.md
# Model card: bellium/micro-nn/tone-curve:v0

- Task and bounded inputs: parameterizes a smooth exposure curve from the global
  luminance distribution of one image. Eight inputs (`p5`, `p25`, `p50`, `p75`,
  `p95`, `mean`, `std`, `dyn_range`) and four outputs (`gamma_norm`,
  `lift`, `gain`, `contrast_pivot`).
- Public API and version: `bellium.micro_nn.tone_curve.extract_tone_features(image)`
  and `micro_tone_adjust(image, *, model=None) -> SpecialistResult`. v0.
- Authority mode: `consultative`.
- Model / exemplar source revision and license: hand-authored weights in
  `models/micro_nn/tone-curve/v0.json`, Apache-2.0, no third-party model or
  dataset.
- Training data and seeds: none recorded. Shapes `[8, 8, 4]`, activations
  `relu` then `sigmoid`, 108 parameters counted from the declared shapes. The file
  records `heldout_accuracy: 0.965` from the authoring run; it is declared rather
  than reproduced here, and no corpus or seed is published with it.
- Independent evaluation data: `benchmarks/manifests/magick-replacement-v1.json`
  (2026-09-19), tracks `gradient`, `step_edge`, `lineart`.
- Deterministic baseline: `bellium.editing.tones.contrast_stretch`,
  `histogram_equalize` and `auto_gamma`; those three modes are reachable without
  the network.
- Quality, false agreements and abstention: the measured curves stay close across the
  three tracks (gamma 1.434 to 1.464, lift 0.051 to 0.053, gain 1.076 to 1.083,
  pivot 0.611 to 0.630), which is the intended stability. There is no abstention
  path and no published comparison against the three deterministic tone modes, so
  the network's marginal value is not established.
- Hardware and software versions: Windows workstation, CPython 3.14.
- Latency, cold/warm policy, repetitions: whole-run wall times from the manifest,
  `micro_tone_ms` 1.95 (`step_edge`), 2.01 (`gradient`) and 1.64 (`lineart`).
  Single authored runs.
- RAM / VRAM: not measured.
- Escalation and invalid input behavior: the result carries the authority mode and
  the curve parameters, and the caller applies the curve; nothing is written to disk.
- Known limitations: the micro tier is not covered by the nano size contract, so the
  model has no declared byte budget and no device verification. No perceptual study
  shows the curve is better than a carefully chosen deterministic stretch.
- Evidence paths and reproduction command:
  `python -m pytest tests/test_magick.py -q` and
  `python scripts/benchmark_magick_replacement.py`.

*** Add File: C:\BelliumAI\Bellium-AI\docs\model-cards\adaptive-threshold-v0.md
# Model card: bellium/knn/adaptive-threshold:v0

- Task and bounded inputs: names the binarization strategy for one image, choosing
  between a global Otsu threshold and a local adaptive window from compact image
  descriptors.
- Public API and version: `bellium.knn.adaptive_threshold.extract_threshold_features(
  image)` and `knn_threshold(image, *, patterns=None) -> SpecialistResult`. v0.
- Authority mode: `consultative`. The specialist names a strategy; the mask is
  produced by `bellium.editing.morphology`.
- Model / exemplar source revision and license: authored exemplar set in
  `models/knn/visual/threshold-patterns-v0.json` (5 patterns, including
  `clean_text`), Apache-2.0, no external corpus.
- Training data and seeds: none. The memory is loaded at call time and can be
  replaced through `patterns`.
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
- Escalation and invalid input behavior: abstention carries the features; a non
  abstaining result carries the chosen strategy and the authority mode.
- Known limitations: 5 authored exemplars, no measured strategy accuracy, and no
  evaluation on real scanned documents.
- Evidence paths and reproduction command:
  `python -m pytest tests/test_magick.py -q`.

*** Add File: C:\BelliumAI\Bellium-AI\docs\model-cards\hash-matcher-v0.md
# Model card: bellium/knn/hash-matcher:v0

- Task and bounded inputs: retrieves the nearest known image from a caller-supplied
  gallery of perceptual hashes. The query is a 64-bit hash and the gallery entries
  carry their own hashes; the match is decided by Hamming distance.
- Public API and version: `bellium.knn.hash_matcher.match_hash_knn(query_hash,
  gallery, *, k=1, max_distance=10) -> SpecialistResult`. v0. The hashes themselves
  come from `bellium.magick.hashes` (`compute_ahash`, `compute_dhash`,
  `compute_phash`, `hash_to_hex`).
- Authority mode: `consultative`.
- Model / exemplar source revision and license: no model file in this repository. The
  gallery is entirely supplied by the caller, so no third-party data is bundled.
- Training data and seeds: none; hashes are deterministic functions of the input
  image.
- Independent evaluation data: none published for the retrieval itself.
  `tests/test_fastimage.py` (`test_perceptual_hashes_and_knn_matching`) covers the
  hash functions and a synthetic match.
- Deterministic baseline: exact Hamming comparison over the gallery. The k-NN adds a
  ranked shortlist and a declared `max_distance` bound rather than a different
  metric.
- Quality, false agreements and abstention: an empty gallery abstains with reason
  `empty_gallery` and echoes the query hash in hexadecimal. A near-duplicate and a
  different image are separated only by the declared distance bound, so the caller
  owns that threshold; no false-positive rate is published.
- Hardware and software versions: Windows workstation, CPython 3.14.
- Latency, cold/warm policy, repetitions: not measured.
- RAM / VRAM: not measured.
- Escalation and invalid input behavior: abstention on an empty gallery; otherwise
  the ranked candidates, their distances and the bound are returned with the
  authority mode.
- Known limitations: perceptual hashes are sensitive to crop, rotation and colour
  changes, and nothing here measures that sensitivity. The distance bound is the
  only guard against a false agreement.
- Evidence paths and reproduction command:
  `python -m pytest tests/test_fastimage.py -q`.

*** Add File: C:\BelliumAI\Bellium-AI\docs\model-cards\raster-to-svg-v0.md
# Model card: bellium/hybrid/raster-to-svg:v0

- Task and bounded inputs: traces a flat-colour raster into SVG paths. The input is an
  in-memory image; the output is SVG text plus the region report. Quantization,
  pixel-square contour tracing and the region routing all run in
  `bellium.editing.vectorize`.
- Public API and version: `bellium.specialists.vectorize.raster_to_svg(query) ->
  SpecialistResult`, with the underlying
  `bellium.editing.vectorize.vectorize_regions`, `rasterize_regions` and
  `regions_to_svg`. v0.
- Authority mode: `consultative`. Nothing is written to disk; the SVG comes back as
  data.
- Model / exemplar source revision and license: authored exemplar set in
  `models/knn/visual/vector-regions-v0.json` (7 exemplars), Apache-2.0. The tracing
  fixture is original geometric clip-art authored for this repository.
- Training data and seeds: none. The k-NN memory is loaded at call time and the
  classifier is `bellium.knn.vector_region.classify_vector_region`.
- Independent evaluation data: `tests/test_vectorize.py`; the fixture is the
  pelican-and-bicycle clip-art, which is deliberately not a prompt-to-image or
  photographic benchmark.
- Deterministic baseline: the palette is built with median cut and mapped with
  `quantize_nearest`; the contour trace is exact pixel-square geometry, so the same
  input yields the same paths.
- Quality, false agreements and abstention: a photographic region abstains instead of
  being traced, and an empty or unusable region list abstains rather than emitting an
  empty document. The classifier splits flat fills from thin strokes. No coverage or
  fidelity metric is published for the traced paths, so the geometric error of the
  output is not bounded.
- Hardware and software versions: Windows workstation, CPython 3.14.
- Latency, cold/warm policy, repetitions: not measured.
- RAM / VRAM: not measured.
- Escalation and invalid input behavior: a missing `image` raises, a photographic
  colour count abstains, and the result carries the authority mode and the region
  report.
- Known limitations: the specialist is not a pixel tracer and not a CAD kernel; it
  refuses gradients and photographic content, and the 7-exemplar memory is authored
  rather than measured.
- Evidence paths and reproduction command:
  `python -m pytest tests/test_vectorize.py -q`.

