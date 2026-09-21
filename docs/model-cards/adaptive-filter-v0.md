# Model card: bellium/nano-nn/adaptive-filter:v0

- Task and bounded inputs: decides how much to sharpen structural contours and how
  much to smooth noisy texture in one pass over an in-memory image. Four inputs
  (`grad_mag`, `local_var`, `hf_energy`, `noise_proxy`) and two outputs,
  `sharpen_weight` and `smooth_weight`, produced by a softmax.
- Public API and version: `bellium.nano_nn.adaptive_filter.nano_adaptive_filter(
  image, *, model=None) -> SpecialistResult`. v0.
- Authority mode: `consultative`.
- Model / exemplar source revision and license: hand-authored weights in
  `models/nano_nn/adaptive-filter/v0.json`, MIT, no third-party model or
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
