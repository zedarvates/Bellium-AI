# Small-learning gate - 9 October 2026

This increment improves the existing memory k-NN and stdlib micro-NN trainers.
Published model weights are unchanged. No new runtime dependency is required.

## Memory retrieval

The old ASCII tokenizer split accented words and rejected scripts without ASCII
letters. Tokens now preserve Unicode letters/numbers after casefold and NFC
normalization. Accents remain significant: `mémoire` is not `memoire`.
Retrieval does not transliterate, infer synonyms, stem words or segment Chinese
sentences into words. Combining marks that NFC does not compose are delimiters.

File-loaded and caller-supplied memories share the same validation, including
evidence classes, non-empty terms, forbidden payload fields and unique ids.
`k` must be a positive integer. Domain isolation and the minimum of two supported
neighbors remain in force.

Top-k selection uses a heap rather than materializing and sorting every matching
candidate. Scores and the abstention threshold are unchanged. Ties use descending
weighted score, descending raw similarity, then ascending id; file order no longer
chooses the winning memory. This can change the top id for equal-score fixtures.
A randomized 200-record test compares results with a full-sort reference at four
values of k. Memory loading/validation still scans and copies the full record list;
this gate does not establish end-to-end latency or RAM savings.

The subsequent [prepared-memory gate](MEMORY_PREPARED_GATE.md) adds an optional
immutable snapshot and measures repeated-query latency and Python allocations.
The list path described above continues to validate each query.

Five authored NFC-memory/NFD-query cases cover English, French, Spanish, Greek
and Chinese terms. All five now retrieve two neighbors at similarity 1.0. The
previous ASCII tokenizer reaches the support threshold only on the English case.
These are normalization fixtures, not multilingual retrieval-quality evaluation.

## Micro-NN training

Both trainers reject invalid layers, feature dimensions, non-finite numbers,
classifier labels outside the output range and invalid training controls before
training. Scalar regression requires exactly one output. Finite arithmetic checks
reject overflowing preactivations and parameter updates. Classifier samples are
copied before shuffling, making repeated same-seed calls reproducible without
reordering caller data.

Optional controls on both `train_classifier` and `train_regressor`:

- `validation_samples`: a separate non-empty split, checked before training.
- `patience`: positive epoch count without sufficient validation improvement;
  requires validation samples. Omit it to run all epochs with best-weight selection.
- `min_delta`: non-negative improvement required to reset patience. The lowest
  observed loss always owns the saved weights, including smaller improvements.

Classification monitors stable cross-entropy. Regression monitors MSE in scaled
target units. The returned model restores the best weights and includes `training`
metadata: epoch counts, best epoch/loss, loss name and stopping controls. Validation
selects weights and must not double as the final test split. With no validation,
fixed-epoch behavior remains; both trainer outputs matched the HEAD implementations
exactly on a bounded 20-epoch compatibility probe.

```python
from bellium.micro_nn.train import train_classifier

model = train_classifier(
    training_samples,
    layer_sizes=[2, 6, 2],
    validation_samples=validation_samples,
    epochs=80,
    patience=8,
    min_delta=0.0001,
    lr=0.05,
    seed=7,
)
```

## Measured synthetic result

`scripts/benchmark_small_learning.py` uses a linear two-feature classification
rule, 15% random training-label noise, and independently generated train,
validation and final-test splits (120/80/200 samples, seeds 101/102/103).
The network has 32 parameters. Initialization seeds are 3, 7 and 11; these share
the same data splits and do not constitute three independent datasets.

| Initialization | Fixed 80 epochs: test accuracy | Validation: test accuracy | Epochs run / best |
| --- | --- | --- | --- |
| 3 | 79.5% | 98.5% | 17 / 9 |
| 7 | 91.5% | 91.5% | 19 / 11 |
| 11 | 88.0% | 90.0% | 9 / 1 |
| Mean | 86.3% | 93.3% | 15 epochs run |

This is a fixed-epoch versus validation-selection comparison. The label-generating
deterministic rule is perfect by construction and remains the reference. This
experiment does not justify replacing that rule or claim improvements on existing
shipped specialists, real data or physical estimates. Regression is covered by
correctness tests; no regression-quality benchmark is claimed here.

Reproduce with:

```powershell
python scripts/benchmark_small_learning.py --output benchmarks/results/small-learning-2026-10-09.json
```

The JSON report contains all per-seed results and local elapsed times. Timing is
one run per configuration; no portable speedup claim follows from it.

## Verification and next gate

On Windows / Python 3.13.5: 1,102 pytest tests pass, 15 internal suites pass,
14 example integration tests pass, Ruff passes and `git diff --check` passes.
The 51 new test cases include invalid inputs, reproducibility, early stopping,
best-weight restoration, overflow rejection, Unicode normalization and top-k
equivalence. No Python 3.10/3.14 or non-Windows run was performed in this gate.

Next: evaluate validation-based training on existing specialist datasets with
separate train/validation/test splits and declared acceptance thresholds before
changing any shipped weights; measure memory retrieval at larger corpus sizes.
