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
