# Native calibration stability gate - 10 October 2026

Temperature fitting was repeated on ten fresh split pairs per native classifier.
The result is more nuanced than the preceding single-split rejection: some fits
help, and inpainting improves on pooled data, but neither procedure meets the
predeclared eight-of-ten stability requirement. Defaults, weights and thresholds
remain unchanged; no temperature is selected from evaluation results.

## Acceptance and evidence integrity

The study exposed a concrete weakness in `promotion_gate`: comparisons against NaN
can all be false, allowing otherwise favorable evidence to pass without a valid
loss or error value. The gate now rejects non-finite/boolean metrics, inconsistent
sample/support/acceptance denominators and invalid per-class arrays before making
quality comparisons. Per-class losses must also reconstruct the overall loss.
This changes malformed-evidence handling, not the existing quality thresholds.

Pooling validates reliability-bin boundaries, counts, confidence/accuracy values,
and their agreement with reported accuracy and ECE. Pooled ECE is recomputed from
pooled bin sufficient statistics. Averaging absolute per-split ECE gaps is not the
same operation: a test has two split errors of0.2 that cancel to pooled ECE0.0.

The summary requires all ten unique predeclared pair identifiers/seeds and
recomputes every individual quality gate. A favorable pooled result cannot hide
too few passing splits, and a manually altered gate result is rejected.

## Frozen replication protocol

The [protocol](../benchmarks/protocols/native-calibration-stability-v1.md) was saved
before execution. Each model uses ten calibration/evaluation seed pairs:
2026101000+2*i and2026101001+2*i, i=0..9, with1,500 calibration and2,500 evaluation
samples per pair. Temperature bounds[0.1,10],60 search iterations, cutoff0.55,
ten ECE bins and all preceding quality criteria remain fixed.

Each pair must pass the preceding complete quality gate. The fitting procedure
must pass at least eight pairs and the same quality gate on pooled metrics.
This evaluates a procedure, not a deployable single temperature. The median
temperature below is descriptive and is not chosen for inference.

The study audits feature uniqueness/zero overlap within and across all new splits
per model and known reference generation seeds, including the preceding negative
calibration/evaluation split. There are40,000 new generated samples per model
(15,000 calibration +25,000 evaluation); both models share their existing synthetic
generator and trained weights across repetitions. This is not field annotation or
an estimate across independently trained models.

## Observed results

| Model | Passing pairs | Pooled NLL before / after | Relative NLL improvement | Pooled quality gate | Procedure stability |
| --- | --- | --- | --- | --- | --- |
| Tool router | 2/10 | 0.020811 /0.020674 | 0.66% | Reject: below2% benefit | Reject |
| Inpaint router | 6/10 | 0.041126 /0.040038 | 2.65% | Pass | Reject: fewer than8 passes |

Tool-router NLL improves in six pairs; only two satisfy all criteria. Temperature
range0.9327–1.3040, median1.1372. Eight pairs fail the minimum NLL benefit, and one
also fails Brier non-regression. Pooled ECE improves0.003158 ->0.002347, but the
benefit in pooled NLL remains below the declared threshold.

Inpaint NLL improves in seven pairs; six satisfy all criteria. Temperature
range0.9907–1.5960, median1.2735. Four fail minimum NLL benefit, two fail Brier
non-regression, and one fails ECE tolerance; rejection reasons can overlap.
Pooled ECE improves0.005177 ->0.001976 and pooled Brier0.024803 ->0.024323.
This favorable mean does not erase the split-level instability.

Pooled accepted counts change24961 ->24955 for tools and24919 ->24890 for inpainting
when experimental scores are applied at the unchanged0.55 cutoff. Pooled argmax
accuracies stay99.064% and98.3%. No reference labeling rule or model weight changes.

## Replay and verification

The original study report is preserved. A second report independently replays all
twenty evaluation seeds using the saved temperatures, without fitting again.
It reconstructs every raw/corrected metric exactly, recomputes summaries and checks
actual `predict_mlp` versus `predict_calibrated` decisions:25,000 per model,
50,000 total, zero decision changes. Model file hashes match before, after and
during replay. Replayed samples are not counted as additional independent evidence.

```powershell
python scripts/benchmark_calibration_stability.py --output benchmarks/results/native-calibration-stability-2026-10-10.json
python scripts/benchmark_calibration_stability.py --verify-report benchmarks/results/native-calibration-stability-2026-10-10.json --output benchmarks/results/native-calibration-stability-verified-2026-10-10.json
```

The verifier rejects overwriting the input report. Reports preserve all pair
metrics, reliability bins, fitted artifacts, rejection reasons, protocol/script
hashes and model hashes. No sidecar is saved or automatically applied.

Source verification:1,261 pytest tests pass,15 internal suites,14 example tests,
Ruff and diff --check pass on Windows/Python3.13.5. Thirty-eight new tests cover
malformed evidence, correct pooled denominators/ECE, incomplete or altered runs,
and the distinction between favorable pooled results and procedure stability.

This increment modifies evaluation scripts/tests/docs only; native runtime APIs
and the preceding local dist-calibration wheel remain unchanged. There is no new
wheel build, commit or publication claim for this study. The complete results are
synthetic rule-fidelity evidence, not real-world probability calibration or a
population confidence interval. Further changes need fresh declared evaluation
splits rather than tuning against these observed outcomes.

Next bounded gate: test whether a larger predeclared calibration sample budget
reduces fitting variability for the promising inpaint procedure, with fresh seed
pairs, fixed quality/stability criteria and explicit CPU cost. No rollout follows
from a better mean alone.
