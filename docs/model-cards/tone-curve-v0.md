# Model card: bellium/micro-nn/tone-curve:v0

- Task and bounded inputs: parameterizes a smooth exposure curve from the global
  luminance distribution of one image. Eight inputs (`p5`, `p25`, `p50`, `p75`,
  `p95`, `mean`, `std`, `dyn_range`) and four outputs (`gamma_norm`,
  `lift`, `gain`, `contrast_pivot`).
- Public API and version: `bellium.micro_nn.tone_curve.extract_tone_features(image)`
  and `micro_tone_adjust(image, *, model=None) -> SpecialistResult`. v0.
- Authority mode: `consultative`.
- Model / exemplar source revision and license: hand-authored weights in
  `models/micro_nn/tone-curve/v0.json`, MIT, no third-party model or
  dataset.
- Training data and seeds: none recorded. Shapes `[8, 8, 4]`, activations `relu`
  then `sigmoid`, 108 parameters counted from the declared shapes. The file records
  `heldout_accuracy: 0.965` from the authoring run; it is declared rather than
  reproduced here, and no corpus or seed is published with it.
- Independent evaluation data: `benchmarks/manifests/magick-replacement-v1.json`
  (2026-09-19), tracks `gradient`, `step_edge`, `lineart`.
- Deterministic baseline: `bellium.editing.tones.contrast_stretch`,
  `histogram_equalize` and `auto_gamma`; those three modes are reachable without
  the network.
- Quality, false agreements and abstention: the measured curves stay close across the
  three tracks (gamma 1.434 to 1.464, lift 0.051 to 0.053, gain 1.076 to 1.083,
  pivot 0.611 to 0.630), which is the intended stability. There is no abstention path
  and no published comparison against the three deterministic tone modes, so the
  network's marginal value is not established.
- Hardware and software versions: Windows workstation, CPython 3.14.
- Latency, cold/warm policy, repetitions: whole-run wall times from the manifest,
  `micro_tone_ms` 1.95 (`step_edge`), 2.01 (`gradient`) and 1.64 (`lineart`).
  Single authored runs.
- RAM / VRAM: not measured.
- Escalation and invalid input behavior: the result carries the authority mode and the
  curve parameters, and the caller applies the curve; nothing is written to disk.
- Known limitations: the micro tier is not covered by the nano size contract, so the
  model has no declared byte budget and no device verification. No perceptual study
  shows the curve is better than a carefully chosen deterministic stretch.
- Evidence paths and reproduction command:
  `python -m pytest tests/test_magick.py -q` and
  `python scripts/benchmark_magick_replacement.py`.
