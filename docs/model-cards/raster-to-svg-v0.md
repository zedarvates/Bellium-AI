# Model card: bellium/hybrid/raster-to-svg:v0

- Task and bounded inputs: traces flat-colour raster art into SVG paths. The input is
  an in-memory image; the output is SVG text plus the region report. Quantization,
  pixel-square contour tracing and the region routing live in
  `bellium.editing.vectorize`, and the routing decision is a k-NN.
- Public API and version: `bellium.specialists.vectorize.raster_to_svg(query) ->
  SpecialistResult`, over `bellium.editing.vectorize.vectorize_regions`,
  `rasterize_regions` and `regions_to_svg`. v0. Caller options are `max_colors`
  (default 8), `min_area` (4), `epsilon` (0.6), `skip_background` (true) and
  `max_regions` (64).
- Authority mode: `consultative`. Nothing is written to disk; the SVG comes back as
  data and the result is not certified.
- Model / exemplar source revision and license: authored exemplar set in
  `models/knn/visual/vector-regions-v0.json` (7 exemplars), MIT. The tracing
  fixture is original geometric clip-art authored for this repository, not a
  third-party image.
- Training data and seeds: none. The k-NN memory is loaded at call time and the
  classifier is `bellium.knn.vector_region.classify_vector_region`.
- Independent evaluation data: `tests/test_vectorize.py`. The pelican-and-bicycle
  fixture is deliberately original clip-art, so it is not a prompt-to-image or
  photographic benchmark.
- Deterministic baseline: the palette is built with median cut and mapped with
  `quantize_nearest`; the contour trace is exact pixel-square geometry, so the same
  input yields the same paths.
- Quality, false agreements and abstention: two abstention paths are explicit. A field
  with too many distinct colours abstains with reason `photographic_unique_colors`
  (more than twice `max_colors` or the 16-colour floor, and more than 20 % of the
  pixels unique). Otherwise the k-NN drops every region whose nearest exemplar is
  `photo_texture` or whose similarity falls under `MIN_SIMILARITY`, reporting
  `region_not_vector_safe`, and if nothing survives the call abstains with
  `no_vector_safe_regions`. No coverage or fidelity metric is published for the
  traced paths, so the geometric error of the output is not bounded.
- Hardware and software versions: Windows workstation, CPython 3.14.
- Latency, cold/warm policy, repetitions: not measured.
- RAM / VRAM: not measured.
- Escalation and invalid input behavior: a missing `image` raises `ValueError`;
  photographic or noisy fields abstain instead of being traced; the accepted result
  carries the SVG, the region counts, the palette and a raster preview.
- Known limitations: not a pixel tracer and not a CAD kernel, no gradient or
  photographic support, and the 7-exemplar memory is authored rather than measured.
  The code notes that a photorealistic prompt-to-image test would still need a real
  image model, which this specialist is not.
- Evidence paths and reproduction command:
  `python -m pytest tests/test_vectorize.py -q`.
