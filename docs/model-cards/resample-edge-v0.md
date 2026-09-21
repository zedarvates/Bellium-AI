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
  `models/nano_nn/resample-edge/v0.json` in this repository, MIT. No
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
