Model card: bellium/knn/integration-method:v0

- Task: name one of the four published integrators for a normal field before spending the time to
  integrate it.
- Authority: consultative and inert. integrated, certified and pixels_changed stay false, false and
  0: no height is produced and no map is changed.
- Inputs: normals, optional mask, min_cosine, neighbours and an optional memory object. Undocumented
  query keys are refused rather than ignored.
- Features, five and deterministic, all in [0, 1]: direction_focus, slope_uniformity,
  divergence_ratio, radial_alignment, flat_field. They are read from the slopes, so the answer costs
  no integration.
- Memory: models/knn/integration/method-selection-v0.json, 16 exemplars. The four controlled surfaces
  at four declared noise levels (sigma 0, 0.02, 0.05, 0.1), built at seed 1, labelled with the method
  that measured best, with all four relative errors stored beside each label. 12 labels are
  cumulative-average, 3 are least-squares and 1 is cumulative-vertical.
- Tie-break: equal errors go to the cheaper method, by the measured p50 latency of the benchmark
  (cumulative-vertical 1.14 ms, cumulative-average 1.17 ms, cumulative 1.20 ms, least-squares
  242.2 ms). An exact ramp is therefore labelled cumulative-vertical, not least-squares, and the same
  order defines the label of a scored field.
- Held-out measurement: 32 fields at seeds 2 and 3, never in the memory. The recommendation names the
  best method in 31 of 32 fields (96.9 %) at a mean relative error of 0.0635, against 0.0634 for an
  oracle choosing per field, 0.0764 for the published rule (25 of 32) and 0.0897 for the fixed
  default (23 of 32). The single miss is an oscillation at noise 0.1, where the two candidates
  measure 0.2749 and 0.2728.
- Where it earns its keep: on a perturbed ramp the rule answers cumulative-vertical (0.0057) where
  least-squares measures 0.0038, because a rule cannot see noise in an otherwise constant slope. The
  disagreement is reported as learned_tier_overrides_the_published_rule instead of being silent.
- Baseline: bellium/deterministic/integration-rule:v0 is always returned beside the recommendation,
  and it answers instead when fewer than three exemplars clear the similarity floor (status baseline,
  confidence 0.5).
- Verdicts: ready, or abstain with field_too_small when fewer than four pixels carry a slope.
- Limits: four geometries, one noise model, 32 x 32 fields. cumulative and cumulative-vertical never
  won on a perturbed field here, so the tier carries no evidence about them. Nothing in this card is
  a measurement of a real normal map, and the recommendation is not a quality verdict on the field.
- Evidence: tests/test_integration_method.py, scripts/build_game_memories.py,
  scripts/benchmark_normal_integration.py, docs/NORMAL_INTEGRATION_GATE.md.

