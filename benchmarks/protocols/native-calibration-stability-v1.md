# Native calibration stability v1 - 10 October 2026

Predeclared before execution: repeat the previous temperature-scaling gate on
both distributed native classifiers, without training, changing thresholds,
enabling a correction or selecting a best evaluation seed.

- Models: tool-router and inpaint-router, their current distributed v0 weights.
- Ten split pairs per model; pair i=0..9 uses calibration seed2026101000+2*i and
  evaluation seed2026101001+2*i.
- Each pair: 1,500 labeled calibration and 2,500 labeled evaluation samples from
  the existing trainer generator. Audit zero feature overlap across all new pairs
  and known prior trainer/validation/evaluation generation seeds.
- Fit: unchanged scalar temperature, bounds[0.1,10], 60 search iterations.
- Score cutoff0.55 and ECE ten bins stay fixed. Every pair uses the preceding
  native calibration gate's complete THRESHOLDS, including class support and
  per-class error limits.
- Summarize all ten fits, temperature range, per-pair metric differences,
  pass/rejection counts, worst metric differences and support-weighted pooled
  metrics. Pooled ECE must be recomputed from pooled reliability-bin sufficient
  statistics; averaging per-split ECE is not pooled ECE.
- Stability criterion: at least eight of ten individual gates pass, and pooled
  metrics pass the same preceding quality gate. No parameter is promoted on this
  criterion: it evaluates stability of the fitting procedure, not an individual
  artifact. No choosing a temperature from evaluation results.
- Missing, non-finite or inconsistent evidence must fail before acceptance.
- Persist every pair and model hash; do not keep only favorable runs. Original
  negative results remain evidence and are not part of the new tuning samples.

Scope: generated rule-fidelity labels, shared trained model and generator across
pairs; no independent human/field calibration claim or population confidence
interval. Runtime Windows/Python3.13.5, CPU only. No weights, default scores,
sidecars or publication change.
