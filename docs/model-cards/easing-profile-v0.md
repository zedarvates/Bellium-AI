Model card: bellium/knn/easing-profile:v0

- Task: name the easing curve of a motion series (linear, ease-in, ease-out,
  ease-in-out, overshoot).
- Authority: consultative. pixels_changed stays 0 and no frame is retimed.
- Inputs: family (ui, character, effect) plus either a transition series or seven
  normalised progress samples.
- Measurement: the series is turned into a normalised progress curve that starts at the
  origin, so it can be compared with a published reference curve directly.
- Memory: models/knn/motion/easing-profiles-v0.json, the five published curves per family
  (15 items), rebuilt by scripts/build_game_memories.py. No measurement is stored.
- Decision: the nearest curve must fit within 0.06 RMSE, must beat the runner-up by
  0.015, and must agree with the published threshold rule; otherwise the specialist
  abstains (no_curve_close_enough, ambiguous_easing, baseline_disagrees).
- Measured on held-out clips (see the gate record): 12 of 20 named at zero noise with
  7 abstentions, against 15 of 20 for the published rule. The gap is resolution: with
  fewer than five transitions the seven-point fit is too coarse and the tier abstains
  while the rule still answers.
- Neural candidate: measured and not shipped. A [7, 4, 5] network (57 parameters) trained
  on noisy curves reached 0.477 agreement against 0.658 for the published rule across
  three noise levels, so no weights were written.
- Limits: five named curves only. A custom bezier, a spring or an interrupted curve is
  reported as no close curve instead of being stretched onto one of the five.
- Evidence: tests/test_animation_timing.py, scripts/benchmark_animation_timing.py.
