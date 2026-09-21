# Model card: bellium/knn/palette-match:v0

- Task and bounded inputs: maps a colour, or a whole image, onto a palette supplied
  by the caller. Perceptual distance is the metric, so the mapping does not depend on
  the channel order of the palette.
- Public API and version:
  `bellium.knn.palette_match.match_color_knn(color, palette, k=1) ->
  list[tuple[int, Rgb, float]]` and `quantize_palette_knn(image, palette) ->
  SpecialistResult`. v0.
- Authority mode: `consultative`.
- Model / exemplar source revision and license: no model file and no exemplar
  memory. The metric is `bellium.editing.quantize._perceptual_dist_sq`, part of this
  repository's Apache-2.0 code, and the palette always comes from the caller.
- Training data and seeds: none; this specialist has nothing to train.
- Independent evaluation data: none published. The palette path is covered by
  `tests/test_magick.py` (`test_quantization_and_palette_matching`) on synthetic
  inputs.
- Deterministic baseline: exact nearest-colour search with the same perceptual
  metric and no learned component; the k-NN answer differs only by returning the
  k best candidates with their distances and coverage diagnostics.
- Quality, false agreements and abstention: the distances are exact and reported per
  candidate, so a false agreement can only come from the caller's palette. An empty
  palette raises instead of returning a silent identity mapping. There is no
  abstention path, because the palette defines the full answer space.
- Hardware and software versions: Windows workstation, CPython 3.14.
- Latency, cold/warm policy, repetitions: not measured.
- RAM / VRAM: not measured.
- Escalation and invalid input behavior: `ValueError` on an empty palette; the
  quantized result carries the authority mode and its diagnostics.
- Known limitations: no perceptual or held-out evaluation, no latency record, and no
  dithering: this specialist only decides which palette entry each pixel takes.
- Evidence paths and reproduction command:
  `python -m pytest tests/test_magick.py -q`.
