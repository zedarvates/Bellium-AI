Model card: bellium/nano-nn/frame-phase:v0 (measurement record, no model shipped)

- Task: name the motion phase of one frame transition (hold, build, impact, recover).
- Authority: consultative. The result reports baseline_only and model_shipped false.
- Decision today: the published rule bellium/deterministic/phase-threshold:v0 answers.
  It compares the changed ratio and its slope; it is exact, cheaper and smaller than any
  network that imitates it.
- Measurement: scripts/train_nano_frame_phase.py trains a [7, 4, 4] candidate (52 parameters,
  1591 bytes) on real deltas from generated motion sequences and compares it with the rule
  on 200 held-out transitions. Candidate agreement 0.970 against 1.000 for the rule, so no
  weights were written. The script records the negative result instead of shipping a
  candidate that only imitates a threshold.
- Budget kept ready: 64 parameters and 8192 bytes. A bundled model would be refused above
  that budget, and the code path is exercised in tests with an in-memory model.
- Limits: phases describe motion change, not semantic animation labels. No timing, easing,
  rig or engine curve was evaluated, and the measurement set is synthetic.
- Evidence: tests/test_frame_phase.py, scripts/train_nano_frame_phase.py.
