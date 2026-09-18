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

## P7 — Fast physical-relation estimates

Planned research: bounded local specialists for quick estimates of physical
relations, starting with terrestrial conditions and supporting robotics,
aquaponics, games and simulation previews. These capabilities are not implemented.

Compare three complementary mechanisms against analytical formulas, lookup
tables and reference solvers before choosing a model:

- **k-NN:** retrieve comparable measured or simulated cases and interpolate only
  within a documented domain with sufficient neighbor support.
- **Micro-NN:** small regressors for coupled relations or residual corrections
  where the reference computation is costly enough to justify approximation.
- **Nano-NN:** even smaller, narrowly specialized neural regressors, optionally
  distilled or quantized for constrained devices. Define parameter count, model
  bytes, precision, peak RAM and target-device latency per specialist; these are
  project size labels, not standardized architectures or language models.

| Planned specialist | Inputs and bounded estimates |
| --- | --- |
| `earth-gravity-v0` | Mass, position/altitude and stated Earth model; local gravitational acceleration and weight. Keep the analytical baseline when it is already cheaper and accurate enough. |
| `pressure-relations-v0` | Fluid, density, depth/altitude, temperature and boundary conditions; hydrostatic or atmospheric pressure in separately declared regimes. |
| `humidity-relations-v0` | Temperature, pressure and an explicit humidity measure; relative/absolute humidity and condensation tendency within the calibrated range. |
| `contact-grip-v0` | Material pair, roughness, load, moisture and contact conditions; static/dynamic friction and slip tendency. Model adhesion separately when supported by measurements. |
| `force-balance-v0` | Masses, accelerations, contact geometry and applied forces; bounded estimates of resultant forces and equilibrium residuals. |

Every estimate must carry units, coordinate frame where applicable, assumptions,
valid input range, provenance and calibrated uncertainty. Missing essential
inputs, insufficient neighbors or out-of-domain conditions trigger abstention
or a reference calculation. Visual material appearance alone cannot establish
friction, adhesion or other physical coefficients. Existing deterministic
constraints and simulation authority remain in control.

First gate: define units and reference cases, then benchmark gravity/weight and
pressure baselines. Extend to humidity, contact and coupled forces only with
appropriate reference data. Split evaluation by physical scenario/material,
measure error and worst cases alongside p50/p95 latency, memory and abstention,
and set task-specific acceptance thresholds before training. Retain a neural
candidate only if it provides a measured benefit over the simpler baseline.

## P8 — Image-derived materials, depth and texture repetition

Planned after the first physical-estimation gate, building on P0/P4 visual
primitives. Interpret the request's "Alberto" provisionally as **albedo**.
Produce independently inspectable maps and descriptors, with confidence and
provenance for each output; no material-extraction capability is claimed yet.

| Planned specialist | Separate outputs |
| --- | --- |
| `material-albedo-v0` | Albedo/base-color estimate with illumination and baked shadows distinguished from intrinsic surface color. |
| `material-lighting-v0` | Illumination, shading, shadows, highlights and possible emission, with ambiguity reported. |
| `material-pbr-v0` | Metallic and roughness estimates; normal, height/displacement and ambient-occlusion candidates when supported. Document ranges, color spaces and normal-map convention. |
| `image-depth-v0` | Relative scene depth and confidence. Metric depth requires scale/calibration evidence; scene depth and surface height remain separate. |
| `texture-repeat-xy-v0` | Independent X/Y periods in pixels, confidence, orientation and seam scores; UV repeat counts only for a specified target extent. Report no reliable repetition when appropriate. |
| `texture-tileability-v0` | Horizontal/vertical edge continuity and an optional seamless-tile candidate, preserving the source and exposing the correction. |

Start with classical image processing, autocorrelation/spectral repetition
analysis and patch/material k-NN retrieval. Evaluate micro-NNs for compact map
estimation or correction, and nano-NNs for bounded per-patch decisions when
their measured quality and device cost justify them. A single image can admit
multiple material/lighting/depth explanations: retain uncertainty instead of
treating inferred maps as measured ground truth.

First gate: X/Y repeat detection and tileability on held-out periodic,
nonperiodic, rotated and perspective-distorted textures. Then evaluate albedo
and lighting separation, followed by PBR maps and depth, using licensed paired
captures or controlled renders with known maps. Keep synthetic and real-image
results separate; measure per-map errors, repeat-period errors, seam artifacts,
cross-map alignment, downstream render quality, latency and memory. Physical
coefficients may be linked only through calibrated material evidence from P7.

## P9 — Interactive image editing and direct filters

Planned tools for local image editing, building on P0 cutout/inpainting and P4
asset preparation. This extends the roadmap; an integrated editor is not yet
implemented.

| Planned tool | User-facing behavior |
| --- | --- |
| `magic-eraser-v0` | Brush or select an unwanted object/region, refine the mask, then propose background reconstruction with a before/after preview. Preserve pixels outside the effective mask. |
| `direct-image-filters-v0` | Black-and-white conversion using grayscale, adjustable sepia, invert, brightness, contrast, saturation and tint; optional thresholded black/white as a distinct filter. |
| `local-filter-mask-v0` | Apply a filter to the full image or a selected area, with adjustable strength, mask feathering and protected regions. |
| `image-edit-stack-v0` | Preserve the source, keep ordered editable operations, support undo/redo and before/after comparison, then export a new image. |

Use deterministic pixel operations for direct filters, with documented color
space, clipping and alpha handling. Reuse patch k-NN and classical inpainting
for the magic eraser; compare micro-NN/nano-NN mask refinement, blending or
quality routing only where they improve measured results. Uncertain fills stay
as previews and may abstain or suggest escalation. Existing experimental
inpainting does not establish reliable object removal on photographs.

First gate: grayscale and sepia with adjustable strength, selection masks and
undo/redo; then the magic eraser on small bounded selections. Verify source and
out-of-mask preservation, transparency, operation order and export consistency.
Evaluate erasing seams, texture continuity and failure/abstention on held-out
photographs. Measure preview p50/p95 latency and peak memory by image resolution
on the target device before claiming real-time editing. Preview downsampling
must remain distinct from full-resolution export. Keep artistic color filters
separate from P8 numeric material maps such as normals, depth and roughness.

Status: the deterministic filter core is available in `bellium.filters`
(grayscale, sepia and binary threshold with adjustable strength, optional
mode L/1 masks and alpha preservation) and through the `bellium filter` CLI
command. The interactive editor, mask brush, undo/redo stack and magic eraser
remain unimplemented; no photographic-erasing quality is claimed.

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
