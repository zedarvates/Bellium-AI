# Native tool-router training gate - 9 October 2026

The previous gate added validation-based weight selection to the shared trainers.
This increment evaluates it on an existing specialist and fixes a concrete input
domain mismatch. Distributed weights are preserved.

## Runtime improvement

The tool-router trainer generates five indicators at exactly 0 or 1 and a
continuous criticality score. The public numeric feature schema accepts every
value in [0,1]. This means a value such as wants_mutation=0.499 passes schema
validation while being outside the neural training distribution.

`classify_tool_need` now abstains before loading the model if any of those five
indicators is fractional. It returns label=null, probabilities={}, confidence=0,
reason=outside_training_domain and the unsupported feature names. Criticality
remains continuous, and boolean flags remain supported. No rounding or threshold
coercion invents a known feature value.

The hybrid still applies deterministic policy first. A mutation score >=0.5,
criticality >=0.85 or an existing risk signal triggers the veto before neural
classification. Fractional inputs without a veto receive an abstention. The k-NN
continues to operate under its existing numeric feature contract.

## Evaluation support

`bellium.micro_nn.evaluate.classification_metrics` reports true-row confusion
matrices, class support/recall, overall and balanced accuracy, accepted/abstained
counts, class abstentions, coverage and selective accuracy. Unsupported classes
have recall=null; no accepted predictions means selective_accuracy=null.
Accuracy and recall count all argmax predictions, including those below the
abstention threshold. Selective accuracy counts only accepted predictions.
Softmax output is required. These metrics do not certify probability calibration.

`classification_gate` rejects insufficient class support or recall even when a
global accuracy threshold passes. It also rejects empty, truncated or inconsistent
support evidence. A test demonstrates a majority-class model exceeding 85% overall
accuracy while completely missing its minority class.

## Frozen experiment

The script `scripts/benchmark_tool_router_training.py` uses the existing native
data generator. Training (900, seed 5), validation (300, seed 31) and final test
(1,000, seed 20261009) have no overlapping feature vectors. Initialization seeds
are 5, 17 and 29; the data splits are shared. Topology is [6,8,3], learning rate
0.12, maximum 220 epochs. Validation selection uses patience=20 and min_delta=0.0001.

The main acceptance thresholds are declared before training: overall accuracy
>=95%, recall of every class >=90%, support of every class >=10. Boundary probes
are authored contract checks, with a minimum support of one; they do not estimate
population performance. Their supports are none=1, use_tool=15, escalate=128.

| Initialization | Fixed-epoch final accuracy | Validation-selected final accuracy | Epochs run / best |
| --- | --- | --- | --- |
| 5 | 99.3% | 99.3% | 91 / 71 |
| 17 | 99.5% | 99.0% | 89 / 69 |
| 29 | 98.9% | 99.2% | 48 / 28 |

The existing distributed model reaches 99.3% on the same final split. Seed 5 is
predeclared as the only export candidate; final-test results never choose the
best seed. The deterministic labeling rule is perfect by construction and remains
the reference. Timing is one local run per configuration, not a portable benchmark.

## Negative result retained

All raw networks, including the distributed model, label every authored boundary
probe as escalate. Their overall boundary agreement is 128/144 (88.9%), but
balanced accuracy is only 33.3%: none and use_tool recalls are both zero. The
softmax remains confident rather than abstaining. Early stopping alone does not
solve the input-domain mismatch, so the candidate fails its promotion gate.

After the runtime fix:

- All 144 raw neural queries abstain on their unsupported fractional mutation flag.
- All 128 deterministic escalation cases still escalate in both k-NN and hybrid
  before neural inference.
- All 16 remaining hybrid queries abstain instead of proposing a learned decision.

This records a coverage reduction for unsupported inputs, not an increase in raw
classification accuracy. Supported binary-input behavior remains covered by the
existing tests. The fix does not establish general out-of-distribution detection
for other specialists or calibration near continuous criticality thresholds.

The SHA-256 of distributed tool-router weights remains
`00d45c30fc7392a3f36a6369db2480a11850b734673dd4ecd13c0fbe4161e138`.

## Reproduction and verification

```powershell
python scripts/benchmark_tool_router_training.py --output benchmarks/results/tool-router-training-2026-10-09.json
```

The report includes per-class denominators, stopping epochs, policy checks and
weight hashes. Optional `--candidate-output` writes only after a successful gate;
failure preserves any previous candidate file and exits with code 1. Report and
candidate outputs must differ and cannot target the distributed model file. An
exported candidate would remain separate from the runtime model.

On Windows / Python 3.13.5: 1,153 pytest tests pass, 15 internal suites and 14 example
tests pass, Ruff and diff --check pass. This increment adds 51 tests. No distributed
weight update, external publication or commit was performed.

Next: measure k-NN memory latency and peak Python allocations on larger authored
corpora. Future neural training covering fractional indicators needs its own
declared domain, validation split and fresh final-test probes; these observed
boundary failures must not become a reused test set for a tuned success claim.
