# Model card: bellium/nano-nn/resample-edge:v0

- Task and bounded inputs: subpixel edge refinement while an in-memory RGB image is
  resized. Six inputs per target pixel (the four source corners plus the horizontal
  and vertical gradients) and two outputs (`edge_residual`, `edge_confidence`).
  The model modulates the bilinear baseline; it never invents structure that the
  source pixels do not carry.
- Public API and version: `bellium.nano_nn.resample_edge.nano_resample(image,
  target_height, target_width, *, model=None) -> SpecialistResult`. v0.
- Authority mode: `consultative`. Nothing is written to disk and no caller is
  obliged to accept the refined pixels.
- Model / exemplar source revision and license: hand-authored weights written into
  `models/nano_nn/resample-edge/v0.json` in this repository, Apache-2.0. No
  third-party model or dataset is involved.
- Training data and seeds: none recorded. The declared shapes are `[6, 4, 2]` with
  `relu` then `sigmoid`. The file records `heldout_accuracy: 0.985` from the
  authoring run; that figure is declared, not reproduced independently here, and no
  seed or corpus is published with it.
- Independent evaluation data: `benchmarks/manifests/magick-replacement-v1.json`
  (2026-09-19) over three synthetic tracks: `gradient`, `step_edge`, `lineart`.
- Deterministic baseline: `bellium.editing.resample.resample_bilinear`.
- Quality, false agreements and abstention: on the smooth `gradient` track the
  model refines 0 pixels and reproduces bilinear exactly (`diff_mae` 0.0,
  `diff_ssim` 1.0). It refines 254 pixels on `step_edge` and 2 811 on `lineart`,
  where `diff_mae` is 4.59 and `diff_ssim` 0.9984 against bilinear. Those
  differences are the intended behaviour on structure; they are not yet evidence of
  a perceptual gain. The model has no abstention path of its own.
- Hardware and software versions: Windows workstation, CPython 3.14.
- Latency, cold/warm policy, repetitions: whole-run wall times from the manifest,
  `nano_ms` 10.48 (`gradient`), 14.67 (`step_edge`) and 57.52 (`lineart`)
  against `bilinear_ms` 5.70, 5.79 and 7.70. Single authored runs; no warm or
  repeated policy is recorded.
- RAM / VRAM: not measured.
- Escalation and invalid input behavior: target dimensions are validated by the
  shared `_validate_target_dims` helper before any work starts, and the result
  carries the authority mode and the baseline beside the refined pixels.
- Known limitations: no measured benefit on real photographic upscaling, no device
  verification (`device_verified: false`, null latency and RAM fields), and the
  extra cost on line art is roughly seven times the bilinear pass.
- Evidence paths and reproduction command:
  `python -m pytest tests/test_magick.py -q` and
  `python scripts/benchmark_magick_replacement.py`.

*** Add File: C:\BelliumAI\Bellium-AI\docs\model-cards\resample-knn-v0.md
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
  bilinear baseline. A flat neighborhood matches the `flat_uniform` exemplar, so
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

*** Add File: C:\BelliumAI\Bellium-AI\docs\model-cards\quantize-tone-v0.md
# Model card: bellium/nano-nn/quantize-tone:v0

- Task and bounded inputs: chooses how much Floyd-Steinberg error diffusion to apply
  while an image is quantized to a supplied palette. Four inputs
  (`luma_variance`, `grad_mag`, `dist_centroid`, `luma`) and one output,
  `dither_intensity`.
- Public API and version: `bellium.nano_nn.quantize_tone.nano_quantize(image,
  palette, *, model=None) -> SpecialistResult`. v0.
- Authority mode: `consultative`.
- Model / exemplar source revision and license: hand-authored weights in
  `models/nano_nn/quantize-tone/v0.json`, Apache-2.0. No third-party model or
  dataset.
- Training data and seeds: none recorded. Shapes `[4, 4, 1]` with `relu` then
  `sigmoid`. The file declares `heldout_accuracy: 0.978`; that figure is
  declared by the authoring run and is not reproduced independently here.
- Independent evaluation data: `benchmarks/manifests/magick-replacement-v1.json`
  (2026-09-19), tracks `gradient`, `step_edge`, `lineart`.
- Deterministic baseline: `bellium.editing.quantize.quantize_floyd_steinberg`.
- Quality, false agreements and abstention: the arbiter asks for 0.7856 of diffusion
  on the smooth `gradient` track, 0.9049 on `step_edge` and 0.9999 on `lineart`.
  The intent is to suppress dither noise where the image is smooth and keep full
  diffusion where banding would be visible. No perceptual study backs that intent,
  and the model has no abstention path: a caller that supplies a palette always gets
  a quantized image.
- Hardware and software versions: Windows workstation, CPython 3.14.
- Latency, cold/warm policy, repetitions: whole-run wall times from the manifest,
  `nano_quant_ms` 6.33 (`gradient`), 4.53 (`step_edge`) and 4.57 (`lineart`)
  against `floyd_steinberg_ms` 4.39, 3.03 and 3.03. Single authored runs.
- RAM / VRAM: not measured.
- Escalation and invalid input behavior: an empty palette is refused by the
  underlying quantizer; the result carries the authority mode.
- Known limitations: no device verification (`device_verified: false`, null latency
  and RAM fields), no published banding measurement, and the arbiter adds roughly
  1.5 ms over plain error diffusion.
- Evidence paths and reproduction command:
  `python -m pytest tests/test_magick.py -q` and
  `python scripts/benchmark_magick_replacement.py`.

*** Add File: C:\BelliumAI\Bellium-AI\docs\model-cards\palette-match-v0.md
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
  memory. The metric is
  `bellium.editing.quantize._perceptual_dist_sq`, part of this repository's
  Apache-2.0 code, and the palette always comes from the caller.
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
  quantized result carries the authority mode and the diagnostics.
- Known limitations: no perceptual or held-out evaluation, no latency record, and no
  dithering: this specialist only decides which palette entry each pixel takes.
- Evidence paths and reproduction command:
  `python -m pytest tests/test_magick.py -q`.

*** Add File: C:\BelliumAI\Bellium-AI\docs\model-cards\adaptive-filter-v0.md
# Model card: bellium/nano-nn/adaptive-filter:v0

- Task and bounded inputs: decides how much to sharpen structural contours and how
  much to smooth noisy texture in one pass over an in-memory image. Four inputs
  (`grad_mag`, `local_var`, `hf_energy`, `noise_proxy`) and two outputs,
  `sharpen_weight` and `smooth_weight`, produced by a softmax.
- Public API and version: `bellium.nano_nn.adaptive_filter.nano_adaptive_filter(
  image, *, model=None) -> SpecialistResult`. v0.
- Authority mode: `consultative`.
- Model / exemplar source revision and license: hand-authored weights in
  `models/nano_nn/adaptive-filter/v0.json`, Apache-2.0, no third-party model or
  dataset.
- Training data and seeds: none recorded. Shapes `[4, 4, 2]` with `relu` then
  `softmax`. The declared `heldout_accuracy` is 0.982, taken from the authoring
  run rather than reproduced independently here.
- Independent evaluation data: `benchmarks/manifests/magick-replacement-v1.json`
  (2026-09-19), tracks `gradient`, `step_edge`, `lineart`.
- Deterministic baseline: `bellium.editing.convolve.sharpen` and
  `bellium.editing.convolve.gaussian_blur`, blended by the two weights. With equal
  weights the output is a plain blend of those two deterministic results.
- Quality, false agreements and abstention: the model asks for 0.9385 sharpening and
  0.0615 smoothing on `lineart`, 0.6832 smoothing and 0.3168 sharpening on
  `step_edge`, and 0.6202 sharpening on `gradient`. The split follows the intended
  direction, but no sharpness or noise measurement validates the magnitudes, and
  there is no abstention path.
- Hardware and software versions: Windows workstation, CPython 3.14.
- Latency, cold/warm policy, repetitions: whole-run wall times from the manifest,
  `nano_filter_ms` 26.03 (`gradient`), 25.86 (`step_edge`) and 25.30
  (`lineart`). Single authored runs; the cost is dominated by the two convolutions
  rather than by the network.
- RAM / VRAM: not measured.
- Escalation and invalid input behavior: the result carries the authority mode, the
  two weights and the filtered image; nothing is written to disk.
- Known limitations: no device verification (`device_verified: false`, null latency
  and RAM fields), no perceptual evaluation of the blend, and no published
  comparison against either convolution applied alone.
- Evidence paths and reproduction command:
  `python -m pytest tests/test_magick.py -q` and
  `python scripts/benchmark_magick_replacement.py`.

