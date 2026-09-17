# Asset Factory alpha quality laboratory

This slice makes Bellium useful before a prepared sprite enters the Asset Factory
review flow. It reads real local images, measures their alpha geometry, and returns
advice. It also trains and evaluates dedicated experimental small classifiers.

## Task and authority

Input: a single image with an explicit alpha channel, at least 2×2 pixels, at most
16 megapixels / 32 MiB. Animated images are rejected. RGB-only images need cutout
first. The source byte hash can be checked with `--expected-sha256`.

The six features, ordered by `bellium.alpha-features/v1`, are foreground coverage
(alpha > 128), partial-alpha coverage (15 < alpha < 240), border coverage,
foreground bounding-box area, and absolute horizontal/vertical bounding-box offsets.
Border corners are counted once. Empty mattes have zero bounding-box features.

The deterministic result is `candidate` or `review`. A candidate only has no
detected defect in these matte heuristics. All results keep `production_authorized:
false` and `acted: false`. The optional learned predictions are shadow advice and
cannot replace the rule verdict. Failed decoding, absent alpha or a hash mismatch
returns structured error JSON and exit code 2; a valid advice report returns 0,
including when it asks for review. The exit code is not a production gate.

Useful for: transparent sprites, icons, cutout props and prepared character images.
Excluded: RGB aesthetics, prompt fidelity, missing semantic parts, opaque/tiled
textures, animation, mesh/LOD integrity, licenses and publishing. Hair, glass and
deliberately off-center sprites can trigger review; they need labeled real cases.

The CLI writes a **separate sidecar**. It never extends an existing Asset Factory
GLB manifest, changes its gates, accesses ComfyUI, uploads, or launches a GPU worker.

## Usage

Run from the checkout root with Python 3.10+ and Pillow:

```bash
python -m pip install pillow
python -m bellium.asset_quality inspect sprite.png
python -m bellium.asset_quality inspect sprite.png --output sprite-alpha-review.json
python -m bellium.asset_quality benchmark --output alpha-report.json --models-output alpha-models.json
python -m bellium.asset_quality inspect sprite.png --models alpha-models.json
```

Output files must not exist. No source image or existing report is overwritten.
If a later output write fails after an earlier one succeeds, error JSON lists
`written_outputs` so a caller can recover without rerunning blindly.

Bundle loading verifies its SHA-256, feature order/version, architecture, finite
parameters, support hash and shadow authority. The hash detects accidental changes;
it is not a signature or proof that a dataset label is correct.

## What the models actually are

| Mechanism | Implementation | Learned parameters | Required data |
|---|---|---:|---|
| Rules | Explicit alpha/framing thresholds | 0 | Six measured features |
| k-NN | Five nearest unique training images, RMS feature distance | 0 | Training feature/label ledger |
| Nano | 6 inputs → sigmoid output | 7 | Weights and support ledger |
| Micro | 6 inputs → 8 tanh units → sigmoid output | 65 | Weights and support ledger |

Nano/micro are local project size names. They are neural classifiers, not language
models. Both are trained with class-balanced binary cross-entropy SGD and small L2
regularization in pure Python. They use dedicated visual features. The eleven
imported Botte Secrète agent-task models and their provenance remain unchanged;
matching input dimensions alone is not a valid reason to reuse their weights.

Learned methods require at least three distinct nearby training examples within
RMS distance 0.20. Exact query identities are excluded. They also abstain below
0.80 winning score. These fixed heuristics are uncalibrated. The support ledger is
needed by **both** networks, so their tiny weight files are not their complete
memory footprint. No model is registered, activated or selected automatically.

## Reproducible synthetic evidence

The committed [report](evidence/alpha-qa-synthetic-v1.json) and
[model bundle](evidence/alpha-qa-models-v1.json) use seed 17 and 100 epochs.
The report records code-file SHA-256 values, dataset and model digests, hardware,
feature extraction timing, prediction timing, allocation probes and per-case results.

There are 240 training cases, 60 validation cases, 60 test cases and 12 separate
stress cases. Each original silhouette and its derivatives stay in one partition.
The cases are generated ellipse/rectangle/diamond silhouettes with clean/feathered,
clipped, faded, tiny and blank variants. Labels come from these transformations,
not from running a model. Train and test share transformation families: this is an
implementation comparison, not evidence of generalization to real artwork.

| Method | Correct test answers / 60 | Abstentions | Defects called candidate / 40 | Stress abstentions / 12 |
|---|---:|---:|---:|---:|
| Rules | 60 | 0 | 0 | 0 |
| k-NN | 60 | 0 | 0 | 12 |
| Nano | 38 | 22 | 0 | 12 |
| Micro | 56 | 4 | 0 | 12 |

The twelve stress cases are opaque canvases or scattered alpha. Rules request
review for all twelve. Learned methods abstain on all twelve. The report separates
accuracy among answered cases from coverage and correctness over all cases; an
abstention is never counted as a correct classification.

**The simple rules remain the default.** This synthetic task shows no learned
quality improvement. The micro network has higher coverage than the nano in this
run, but no real-asset or RTX 3060 result is claimed. CPU timings include support
search and exclude decode/feature extraction/loading. Allocation probes measure
Python prediction allocations on one case, not total process RAM. There is no
hosted CI claim in this document; check the pull request for current CI evidence.

## Testing and improving on real assets

Create an annotation manifest next to local source files. Every original, crop,
re-encoding and generation variation from the same source must share a `group` and
`split`. Exact decoded-image identities across splits are rejected even if file
encodings differ. Duplicate training images are collapsed. Contradictory labels
for identical images are rejected.

Minimal case structure (the complete file needs both labels in train, validation
and test):

```json
{
  "schema": "bellium.alpha-dataset/v1",
  "provenance": "Human-reviewed sprite matte labels, reviewer and rubric version",
  "cases": [
    {
      "id": "sprite-001-clean",
      "group": "source-sprite-001",
      "split": "train",
      "label": "candidate",
      "path": "images/sprite-001.png",
      "reason": "Complete silhouette, intentional feathered outline"
    }
  ]
}
```

Allowed splits are `train`, `validation`, `test`, `ood`; labels are `candidate` and
`review`. Paths must stay under the manifest directory. Source files are read only.
Feature/label ledgers and reports from real datasets remain private/local unless
their owner explicitly chooses to publish them; no uploader is included.

```bash
python -m bellium.asset_quality benchmark --dataset local-labels/dataset.json --output real-alpha-report.json --models-output real-alpha-models.json
```

Only train rows enter fitting or k-NN support. Validation, test and stress metrics
are reported separately. This implementation does not tune thresholds or select
a winner. Freeze the test partition before further improvements; change thresholds
or architecture using training/validation evidence, then run a final untouched test.

## Existing pipeline correction

The inherited inpaint function previously processed requests even when its router
asked for escalation, and could report masked pixels as repaired when there were
no valid exemplar centers. It now returns an unchanged image with zero repairs for
these cases. The pipeline propagates escalation instead of letting a confident
cutout declare the unresolved asset ready. Image/mask sizes and positive odd patch
parameters are checked, and a zero mask preserves original image mode and bytes.

The next slice now implements [context-scored patch matching](INPAINT_CONTEXT_QA.md)
with a frozen spatial-copy comparison. `k_neighbors` controls donor aggregation,
and partial repairs are discarded on abstention. Successful fills receive
`needs_review` instead of inheriting production readiness from cutout confidence.
Unrelated paths without filling keep their earlier verdict behavior. Neither
alpha QA nor contextual similarity establishes semantic repair quality.
