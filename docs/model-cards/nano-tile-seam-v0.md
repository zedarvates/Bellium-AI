Model card: bellium/nano-nn/tile-seam:v0

- Task: bounded seam verdict for one texture axis from six deterministic measures.
- Public API and version: bellium.nano_nn.classify_seam; model models/nano_nn/tile-seam/v0.json.
- Authority: consultative. Abstains below 0.60 confidence. Baseline id:
  bellium/deterministic/seam-threshold:v0.
- Inputs: gap, seam, interior, edge_step, seam_max, hot_rows in [0, 1], in that order.
  A missing or out-of-range measure raises instead of being coerced.
- Architecture and declared budget: layers [6, 4, 2], 38 parameters, hard cap 64 parameters
  and 8192 bytes, precision float-json-weights-v1. A model over budget is refused at load.
- Measured size: 38 parameters, 1816 bytes as indented JSON with LF newlines, 1904 bytes on
  disk under Windows CRLF.
- Training data and seeds: 600 synthetic textures measured with bellium.knn._texture and
  labelled by the authored perceptual rule in scripts/train_nano_tile_seam.py, seed 11.
- Independent evaluation data: 200 further synthetic textures from a separate generator seed.
  Held-out accuracy 0.995 against 0.960 for the deterministic baseline on the same vectors.
- Deterministic baseline: the threshold rule is published beside the model and reported in
  every result as baseline and agrees_with_baseline.
- Quality, false agreements and abstention: synthetic vectors only. No real texture or
  rendered material was evaluated, so the measured advantage is not established on real assets.
- Hardware and software versions: Windows, CPython 3.14, this workstation.
- Latency, cold/warm policy, repetitions: warm in-process prediction p50 0.013 ms and p95
  0.014 ms over 2000 calls; full classify_seam including model load and budget check p50
  0.177 ms and p95 0.259 ms over 50 calls.
- RAM / VRAM: not measured. Target-device latency: not measured, device_verified stays false.
- Escalation and invalid input behavior: out-of-budget models, missing measures and
  out-of-range measures fail closed. Low confidence abstains.
- Known limitations: it decides a seam verdict only. It does not repair, crop or retile,
  and it is not a texture classifier or a material estimator.
- Evidence paths and reproduction command: python scripts/train_nano_tile_seam.py --force,
  tests/test_nano_tile_seam.py.
