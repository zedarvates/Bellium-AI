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
