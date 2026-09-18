# Independent compression evaluation — 16 September 2026

The local gate is complete. All 20 samples reconstruct exactly with all six
existing methods. A single, pre-specified spatial micro-NN candidate was rejected
on development data; it is not in automatic selection or the specialist catalog.

## Corpus and provenance

Seven sources, 20 fixed samples, with source-disjoint development and held-out
groups. The [protocol](../benchmarks/protocols/compression-independent-v1.md)
was written before measurement. The [manifest](../benchmarks/manifests/compression-independent-v1.json)
records source URLs, rights evidence, SHA-256, crops/chunks, split and adaptation.
Asset bytes remain outside the repository and package.

- Four photographic sources from scikit-image v0.25.2: camera and coffee for
  development; astronaut and Chelsea for validation. Rights are documented by
  [scikit-image](https://scikit-image.org/docs/stable/api/skimage.data.html).
- Two [OurAirports](https://ourairports.com/data/) public-domain CSV sources,
  pinned to the same full Git commit: countries for development, regions for validation.
- One local trained Fovea splat asset derived from Rico Cilliers' Horse Statue
  under [Poly Haven CC0 terms](https://polyhaven.com/license), reserved for validation.
  Its training source is a synthetic multiview video, not a physical camera capture.

The splat adapter converts the supported fields to canonical physical float32
records. It normalizes/reorders quaternions, exponentiates log scales, applies
sigmoid to opacity and converts clamped DC colors. Higher-order SH is omitted.
**Exact reconstruction applies to these canonical records, not the complete PLY.**
The full original source is separately retained and hashed. No natural motion
sequence or .fovea comparison is included.

## Results

Automatic selection among the existing codecs chooses delta 10 times, zlib nine
times and micro-NN once. k-NN is never the best method.

The existing byte micro-NN wins on `coffee-2`, a development image crop: **21,174
bytes**, versus 21,787 for the best fixed BLCP method (**2.81% smaller**) and
21,478 for PNG (**1.42% smaller**). This isolated development result does not
generalize: no learned method wins on any of the 12 held-out samples. All six
held-out image crops are smaller as PNG than as automatically selected BLCP.

The experimental spatial neuron uses left/up/upper-left context, Q12 weights
and causal normalized updates. It was compared with fixed left, up, average
and clipped-gradient predictors under the same research envelope. Across the
six development image crops, it produces **67,693 bytes**, versus **66,429** for
the best fixed predictor per crop: **1.90% larger**, with an **8.80%** worst-case
size regression. Median-per-sample times summed over the corpus are about 6.2x
slower to encode and 4.9x slower to decode on this run. Timings are machine- and
load-dependent. The required 1% aggregate improvement failed.

Consequently the new candidate is **rejected**, and its held-out evaluation was
not run. Its exact implementation remains in `bellium.compression.spatial_experiment`
for reproducibility, with a distinct research packet kind. BLCP/1 predictors,
golden vectors, regular image API and registry are unchanged by this experiment.

## Reproduction

```powershell
rtk python scripts/prepare_compression_corpus.py --output <new-corpus-directory> --horse-ply <reviewed-horse-ply>
rtk python scripts/evaluate_compression_corpus.py --corpus <corpus-directory> --output <results.json>
```

Preparation performs explicit downloads and requires Pillow. It resolves a fresh
OurAirports commit; preserve the original corpus for exact replay of this report.
Evaluation is offline and verifies the manifest and file digests. The recorded
OurAirports URLs are pinned; the source images use a release tag plus recorded
byte fingerprints. Changing source bytes invalidates the saved corpus.

Each existing method and eligible candidate is measured three times; decode is
checked byte-for-byte every time. Source/code hashes and the protocol hash are
included in the JSON report. The evaluator rejects renamed copies of the same
source crossing the split, unknown paths and changed source bytes.

Tests cover codec roundtrips, causality, alpha, allocation limits, manifest
integrity/split isolation and the explicit PLY adaptation. Results and package
validation for this run are retained in the local
`C:\BelliumAI\compression-independent-2026-09-16` directory.

## Remaining evidence

This is a small, bounded corpus, not population-level or production evidence.
Peak RAM, target-device latency, full SH preservation, natural motion and live
Godot/Zig integration remain unmeasured. Learned-codec promotion is not justified.
Future tuning must use development data and new source groups for a fresh
independent gate; the baseline results on this held-out set are now visible.
