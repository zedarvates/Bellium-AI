# Bellium AI roadmap

Priority goes to small specialists that solve recurring needs in our existing projects.

## P0 — Shared visual primitives

### Selected-zone fill / compact inpainting
Start with deterministic and patch k-NN baselines. Explore a tiny NN for blending/correction and a confidence model that escalates difficult masks to a heavier image model.

### Image cutout
Foreground segmentation, alpha refinement, automatic crop and uncertain-edge confidence for characters, objects, plants, tools, sprites and assets.

### White / transparent background normalization
Background replacement, centering, scale/margin normalization, rotation, exposure and rejection of unsafe transformations. Prefer deterministic CV plus tiny classifiers where sufficient.

### Asset quality
Blur, clipping, compression damage, contamination, poor crop and duplicate/near-duplicate detection. Try embeddings+k-NN before training another network.

#### Implemented experimental slice — alpha matte QA

- Offline `python -m bellium.asset_quality inspect`: actual image bytes, SHA-256,
  six versioned alpha features and consultative JSON sidecar; no source mutation.
- Rules, k-NN, nano 6→1 and micro 6→8→1 compared on group-disjoint fixtures.
- Local annotated-dataset loader, duplicate/split leakage checks, reproducible
  model bundles, abstention and separate validation/test/stress results.
- Inpaint failure propagation: unavailable exemplars and oversized masks cannot
  report a successful repair or let cutout confidence hide an unresolved defect.

See [scope, evidence and commands](ASSET_FACTORY_QA.md). Rules remain the default:
the synthetic run does not show a quality or speed advantage for learned methods.

#### Next bounded slices

1. Label real RGBA sprites from Asset Factory locally, grouping every derivative
   by its source asset/recipe. Freeze train/validation/test identities before fitting.
   Include feathered hair, shadows, intentional off-center sprites and failed cuts.
2. Compare false-candidate rate, abstention, review workload, CPU latency and
   complete working-set memory. Calibrate only on validation data; keep final
   holdout inaccessible to model/threshold selection.
3. Implement genuine context-scored patch k-NN. The inherited inpaint implementation
   currently picks spatially nearest source pixels; `k_neighbors` does not yet
   control its synthesis. Benchmark against that explicit baseline with untouched
   source images, masked-region error and edge continuity.
4. Separate RGB texture QA and near-duplicate retrieval with their own feature
   contracts and annotated families; do not reuse alpha-matte weights for them.
5. Connect reviewed sidecars to the actual Studio/ComfyUI workflow through a
   separate adapter after verifying workflow/node/model/output identities. Keep
   existing GLB manifest gates authoritative; no model-triggered upload.

## P1 — Language preservation and reconstruction

Build provenance-aware specialists for phoneme retrieval/classification, pronunciation similarity, grapheme↔phoneme mapping, cognate/word-form retrieval and prosody. Every result distinguishes `attested`, `reconstructed`, `inferred` and `speculative` evidence.

## P2 — Agent and tool intelligence

Tool/skill/workflow routing, consequence prediction, recurring failure classification and compact memory reranking. Small models propose; deterministic policy and larger models retain authority where required.

## P3 — Robotics and aquaponics

Compact tool/object recognition, visual anomaly/state detection and simulation-first next-action consequence prediction. Emergency stops, collision constraints and forbidden zones remain deterministic.

## P4 — StoryCore, games and asset production

Sprite/texture preparation, panel-safe crops, speech-bubble regions, flat-color helpers, consistency retrieval and bounded NPC behavior classifiers.

## P5 — Audio

Voice activity, speaker changes, recording-quality classification, audio duplicate retrieval, pronunciation validation and segment markers.

## P6 — Micro-LLM laboratory

Use micro-LLMs only when simpler approaches lose enough quality to justify them: structured tool selection, compact intent classification, short metadata extraction, bounded JSON transformations and multilingual label mapping.

## First Bellium-native wave

1. `image-cutout-v0`
2. `white-background-normalizer-v0`
3. `simple-inpaint-router-v0`
4. `patch-knn-inpaint-v0`
5. `asset-quality-v0`
6. `language-phoneme-knn-v0`
7. `pronunciation-similarity-v0`
8. `tool-router-v0`
9. `memory-reranker-v0`
10. `visual-anomaly-v0`

> Bellium AI should become our cabinet of small, sharp intelligences: one bounded specialist for each recurring problem, measured before trusted.
