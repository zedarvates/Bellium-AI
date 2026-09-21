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

## P10 — 2D technical drawing (DAO)

Native vector documents for local drafting, before any 3D CAD kernel. Start with
units, layers and exact 2D primitives, then SVG and DXF round-trips on authored
fixtures. Raster-to-svg remains a pixel tracer and is not this track.

| Planned specialist | Behaviour |
| --- | --- |
| drafting-document-v0 | Y-up document in mm or in; line, polyline, circle, arc, text. |
| drafting-svg-dxf-v0 | Profile SVG and DXF R12 that reconstruct the same document. |
| drafting-ops-v0 | Later: intersection, trim, extend, offset, with abstention on ambiguity. |
| drafting-dims-v0 | Later: cotation, cartouche, orthographic views. 3D CAO is later still. |

First gate: the document and both round-trips, no model. A neural candidate is
not in scope until a deterministic operator loses a measured comparison.

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

## Second Bellium-native wave

1. `grapheme-phoneme-v0`
2. `cognate-retrieval-v0`
3. `prosody-profile-v0`
4. `panel-safe-crop-v0`
5. `speech-bubble-region-v0`

## Third Bellium-native wave

1. `consistency-retrieval-v0`
2. `flat-color-helper-v0`
3. `voice-activity-v0`
4. `recording-quality-v0`
5. `consequence-predictor-v0`

## Compression of images, data streams and Gaussian splats

The first local implementation is available in `bellium.compression`: exact
L/RGB/RGBA pixels, independent byte streams, and finite float32 splat records
with optional explicit-base temporal residuals. Causal k-NN and an adaptive
integer linear micro-NN compete against raw/zlib/delta by measured packet size.
See [compression-v0](model-cards/compression-v0.md) for API, limits and format.

The next gate is an independent asset/stream benchmark, including total bytes,
CPU/memory cost, PNG and the existing .fovea codec. Pretrained nonlinear neural
compression and live consumer integration remain unimplemented. Synthetic
roundtrips alone do not demonstrate superiority on real data.

## Fourth Bellium-native wave: game assets and NPC behaviour

1. `texture-repeat-xy-v0`
2. `texture-tileability-v0`
3. `nano-tile-seam-v0`
4. `texture-tile-fixer-v0`
5. `sprite-anchor-v0`
6. `sprite-frame-prep-v0`
7. `npc-behavior-router-v0` (micro-NN and hybrid)

This wave serves P4 and the first gate of P8. Repeat detection and wrap continuity
are measured on synthetic fixtures built by `scripts/build_game_memories.py`, with
the deterministic baseline published beside both learned tiers. The nano tier now
has a size contract and a measured, if synthetic, edge over its threshold baseline.

Still open in P8: albedo/lighting separation, PBR maps, depth, licensed paired
captures, and any evaluation on real game assets or renders. Still open in P4:
sprite-sheet slicing, atlas packing, engine-import validation and any actor that
executes a proposed behaviour.

## Fifth Bellium-native wave: sprite sheets and action consequences

1. `clip-loop-v0`
2. `frame-phase-v0` (measured, no model shipped)
3. `sprite-sheet-prep-v0`
4. `consequence-precedent-v0`
5. `consequence-predictor-v0` (micro-NN and hybrid)

This wave closes the sprite-sheet gap left open by P4 up to, but not including,
atlas packing and engine import, and it delivers the `consequence-predictor-v0`
item that the third wave listed but never implemented. The frame-phase tier is the
first case where a measured candidate was deliberately not shipped: the published
anomaly-free threshold rule stays the answer.

Still open after this wave: atlas packing, engine-import validation, an animation
timing report (easing and hold lengths), duplicate-frame removal, and the P7
physical-relation gate (units, reference cases, gravity and pressure baselines).

## Sixth Bellium-native wave: P7 first gate delivered

1. `bellium.physics` units and published references (WGS-84 normal gravity with free
   air, ISA atmosphere, hydrostatic and dynamic pressure, ballistic preview helpers)
2. `physical-case-retrieval-v0`
3. `gravity-residual-v0` (nano-NN, four parameters)
4. `atmosphere-model-v0` (micro-NN, 21 parameters)
5. `physical-estimate-v0` (hybrid, reference first)

Thresholds were declared before training from the task the estimates serve, the
comparison runs on a grid that no tier trained on, and every result carries units,
assumptions, valid range, provenance and uncertainty. See
[the gate record](PHYSICAL_ESTIMATES.md).

P7 remains open beyond this gate: humidity, contact-grip and coupled force relations
need reference data that was not available, so they are not implemented. P8 (materials,
depth, texture repetition) still starts from its own first gate.

## Seventh Bellium-native wave: P8 first gate delivered

1. `texture-orientation-v0` (grain direction, angle and period)
2. shift-difference period measurement with bilinear sampling, shared by the orientation
   scan and the axis tests
3. perspective guard in `texture-repeat-xy-v0`, so a warped texture is never reported as
   one clean period
4. `scripts/benchmark_texture_repeat.py` and its held-out report

The gate is measured, not assumed: no held-out case invented a period, rotated grains were
named correctly in four of four cases with a worst angle error of 5 degrees, and the
remaining limitation is repeat-memory coverage for tiled textures (the orientation
specialist still names them). See [the gate record](TEXTURE_REPEAT_GATE.md).

P8 remains open beyond this gate: albedo and lighting separation, PBR maps and depth all
need paired captures or controlled renders with known maps, which this wave did not
produce. Atlas packing, engine-import validation and duplicate-frame removal remain open
in P4; the animation timing report (easing and hold lengths) is still missing.

## Eighth Bellium-native wave: animation timing delivered

1. `bellium/knn/_motion.py`: progress curves, relative hold runs, peak, spacing and
   duplicate mismatches
2. `bellium/knn/easing-profile:v0` (exemplar retrieval over the published curves)
3. `bellium/hybrid/animation-timing:v0` (holds, peak, easing, duplicates, warnings)
4. the timing section inside `sprite-sheet-prep:v0`
5. `scripts/benchmark_animation_timing.py` and its held-out report

Measured: duplicates 45 of 45 across three noise levels, hold totals matching 16 of 20
clean clips, p50 timing 0.33 ms. The easing retrieval abstains when a clip is too short
for a curve fit while the published rule still answers, and a 57-parameter neural
candidate was measured and not shipped because the rule was better under noise. See
[the gate record](ANIMATION_TIMING_GATE.md).

Still open in P4: atlas packing, engine-import validation and any bounded actor that
executes a timing decision. Still open in P8: albedo and lighting separation, PBR maps and
depth, all of which need paired captures or controlled renders with known maps.

## Ninth Bellium-native wave: P8 second gate delivered

1. `bellium/material/controlled.py`: controlled renders with known albedo and
   illumination, evaluation only
2. `bellium/material/separation.py`: log-domain separation, identity fallback, evidence
   and the `separate_render` entry point with declared shading priors
3. `bellium/hybrid/albedo-separation:v0`: the consultative specialist, both candidates
   returned, abstention on clipped highlights
4. `scripts/benchmark_material_separation.py` and its report on 50 controlled renders

The gate produced a negative result worth keeping: a single image cannot say whether its
low-frequency variation is shading or albedo. The tool therefore requires a declared
prior, returns both candidates when it has none, and publishes how often the image-only
recommendation is right (76 %). See [the gate record](MATERIAL_SEPARATION_GATE.md).

P8 remains open beyond this gate: colour cast removal, specular highlights,
interreflection, normal and roughness maps, depth, and any evaluation on real captures.
In P4, atlas packing and engine-import validation are still missing.

## Tenth Bellium-native wave: P8 third gate delivered

1. `bellium/material/photometric.py`: known geometries, multi-light capture generation,
   per-pixel least squares and the residual-based trust rule
2. `bellium/hybrid/photometric-normals:v0`: declared lights, per-patch trust, abstention
   when the Lambertian model fails
3. `scripts/benchmark_photometric_normals.py` and its report on 36 controlled cases

Measured: the fit is exact on lit Lambertian surfaces (0.00 degrees on a plane and a
wavy surface), the residual gate isolates a subset at 0.0094 degrees against 7.86 degrees
for the rejected patches, and the whole set abstains in 10 of 12 strongly specular cases.
A 38-parameter neural gate was measured and not shipped because the published rule was
better at equal coverage. See [the gate record](PHOTOMETRIC_NORMALS_GATE.md).

P8 remains open beyond this gate: colour casts, interreflection, roughness, height
integration, coloured or uncalibrated lights, and any evaluation on real captures or
engine imports. In P4, atlas packing and engine-import validation are still missing.

## Eleventh Bellium-native wave: P9 first gate delivered

1. `bellium/editing/filters.py`: seven deterministic filters with adjustable strength
2. `bellium/editing/selection.py`: rectangle, ellipse, polygon and brush masks, feather
   and masked blending that preserves uncovered pixels byte for byte
3. `bellium/editing/stack.py`: ordered replayable steps with undo, redo, before/after
   and recipe export
4. `bellium/hybrid/direct-filters:v0` and `bellium/hybrid/magic-eraser:v0`
5. `scripts/benchmark_editing.py`: latency by resolution plus five preservation
   invariants, all holding

The measured latency is the honest headline: 4 to 18 ms per operation at 64 x 64 and 34
to 163 ms at 192 x 192 in pure Python, so nothing here is interactive yet. See
[the gate record](EDITING_GATE.md).

P9 remains open beyond this gate: the magic eraser on harder selections, blend modes,
layer compositing, curves and levels, a GPU or array path for interactivity, and
comparison with a neural mask refinement only where it measurably helps.

## Twelfth Bellium-native wave: the interactive path, measured

1. three interchangeable filter backends (vectorised, lookup tables, plain) kept
   pixel-identical by a parametrised test
2. area-averaged previews with their scale, size and preview-only status
3. `PreviewSession`: one conversion of the source, then repeated previews in array space
4. `scripts/benchmark_editing_interactive.py` with a budget declared before the run

Measured: per-operation speedups of 1.5 to 2.9 times from the vectorised path, a cold
one-shot preview of a 512 x 512 source at about 0.5 s, and repeated session previews at
14.7 ms p95 against a 33 ms budget. The honest conclusion is that interactivity comes
from keeping the array representation across a session, not from tuning filters. See
[the interactive record](EDITING_INTERACTIVE_GATE.md).

P9 remains open beyond this gate: blend modes, layers, curves and levels, GPU execution,
multi-threaded editing, and the magic eraser on harder selections.

## P4 gate: atlas packing delivered

1. `bellium/atlas/packing.py`: bottom-left skyline placement, insertion-order and height-sorted
   shelf baselines, a padding gutter, a bounded page count and a plan invariant checker
2. `bellium/hybrid/atlas-packing:v0`: consultative specialist that abstains when frames do not fit
3. `scripts/benchmark_atlas.py` and its report on 10 authored cases x 3 methods

Measured: 0 invariant violations, and the shipped packer completes the mixed and extreme-aspect
cases at 0.9001 and 0.9375 occupancy where the sorted shelf drops 4 and 10 frames and the naive
shelf drops 22 and 24. It ties both baselines on the uniform and near-full cases and costs about
twice their runtime, 0.22 to 1.05 ms for 64 to 400 frames. See
[the gate record](ATLAS_PACKING_GATE.md).

This closes the atlas-packing half of the P4 gap named in the eleventh wave. Engine-import
validation is still missing, and rasterization, rotation, trimming, duplicate-frame removal and
any measurement on real project sheets remain open.

## P4 gate: engine-import validation delivered

1. `models/imports/target-profiles-v0.json`: four declared target contracts, each with its own
   source string, note and conservative limits, plus a caller override path through the same
   strict contract
2. `bellium/imports/targets.py` and `bellium/imports/manifest.py`: findings against a contract and
   JSON manifest mirrors in four shapes, with the Unity bottom-left rect conversion re-checked
3. `bellium/hybrid/engine-import-validation:v0`: ready with warnings or abstain, never a file
4. `scripts/benchmark_import_validation.py` and its report on 21 runs

Measured: 17 ready and 4 abstain over 5 cases x 4 profiles plus one caller-tightened override,
0 manifest violations, manifests of 1.2 to 12.4 KB and 0.32 to 1.8 ms per run. A three-page set
is refused by every single-texture profile and accepted by the multi-page one. See
[the gate record](ENGINE_IMPORT_GATE.md).

No engine was executed, so this closes the P4 gap as a declared contract rather than a measured
round trip. Rasterization, platform texture limits, sprite meshes, nine-patch borders, artist
pivots, trim and rotation paths, duplicate-frame removal and any measurement on real project
sheets remain open.

## P4 gate: atlas rasterization delivered

1. `bellium/atlas/raster.py`: exact page composition in RGB or RGBA with its own invariant checker
2. `bellium/atlas/png.py`: a minimal deterministic PNG writer plus an independent container
   inspector and a decoder for the same subset
3. `bellium/hybrid/atlas-raster:v0`: pack, verify, rasterize, optionally encode; a page that fails
   its check is never returned
4. `scripts/benchmark_atlas_export.py` and its report, cross-checked with Pillow

Measured: 5 cases turn 397312 raw bytes into 184367 PNG bytes (0.464) with 0 invariant violations
and 0 Pillow mismatches; rasterization costs 10.9 to 28.9 ms, encoding 3.5 to 9.7 ms and
verification 5.4 to 15.7 ms. Compression ratios spread from 0.202 to 0.830 by content, which is
what a single-filter encoder does. See [the gate record](ATLAS_RASTER_GATE.md).

The engine round trip stays open: no engine opened these files, platform texture formats and
mipmaps are not produced, and no real project sheet was measured.

## P4 gate: exact duplicate removal delivered

1. `bellium/atlas/dedup.py`: content keys, canonical selection, alias map, duplicate groups, a
   declared per-channel tolerance with measured deltas, mirror matching over the four size-keeping
   transforms, and an independent alias verifier
2. `bellium/hybrid/atlas-dedup:v0`: analyze, verify aliases, pack only the distinct frames,
   rasterize, verify, optionally encode; placements are returned with the pages
3. `scripts/benchmark_atlas_dedup.py` and its report, including a near-miss control

Measured: 9 cases turn 108 declared frames into 59 stored with 86592 bytes saved and 0 violations;
one case turns a two-page set that did not fit into a single complete page, and a mirrored walk of
12 frames stores 4 images that redraw exactly. The near-miss control, whose frames differ by one
channel value, saves nothing, and the same jittered set saves nothing at a declared bound of 1. See
[the gate record](ATLAS_DEDUP_GATE.md).

Still open in P4: a perceptual metric for duplicate removal (the shipped tolerance is a declared
per-channel bound, not a claim about what a viewer notices), quarter turns, deduplication across
atlases, engine-side frame list rewriting, and the engine round trip itself.

## Thirteenth Bellium-native wave: P8 fourth gate delivered

1. `bellium/material/integration.py`: slopes with a declared grazing floor, four integrators, an
   offset-removing error metric and the controlled geometries with known heights
2. `bellium/hybrid/normal-to-height:v0`: the consultative specialist, with convergence reporting,
   the dropped-pixel count and abstention when no slope is recoverable
3. `scripts/benchmark_normal_integration.py` and its report: 16 integrator runs plus the controlled
   capture chain at six declared grazing floors
4. `bellium/knn/integration-method:v0`: five slope features, a 16-exemplar measured memory and the
   published rule returned beside the recommendation

Measured: the row/column average is the best of the four on the controlled set (mean 0.0682,
worst 0.109 at 1.17 ms) while least-squares is exact on a constant slope but not converged on a
dome after 400 sweeps (0.806 at 242 ms), and the vertical alignment is exact on a constant slope
at 1.1 ms. The measurement found two defects and both were fixed: the cone fixture was a
paraboloid, and the row alignment drops the vertical slope of a tilted plane. On the controlled
capture chain, refusing the pixels below a vertical component of 0.1 takes the relative error
from 1.370 to 0.194 while keeping 96 % of them. See [the gate record](NORMAL_INTEGRATION_GATE.md).

The selector is measured held out, on seeds the memory never saw: the k-NN names the best
integrator in 31 of 32 fields at a mean relative error of 0.0635, against 25 of 32 for the published
rule (0.0764) and 23 of 32 for the fixed default (0.0897), with an oracle at 0.0634. Its limit is
the fixture set: four geometries, one noise model, and no evidence about real normal maps.

P8 remains open beyond this gate: colour casts, interreflection, roughness, depth from shading,
multi-view fusion, normal-map convention conversion and any evaluation on real captures or engine
imports. The height is relative by construction and the pixel scale stays a declared input.

## Fourteenth Bellium-native wave: P8 fifth gate delivered

1. `bellium/material/normals_io.py`: 8-bit encode and decode with a mask, the green-channel flip,
   the handedness detector and the displacement quantizer
2. `bellium/material/integration.py`: `integrability`, the cell curl of a slope field with its
   measured bands, returned by every integration
3. `bellium/hybrid/normal-map-convention:v0`: the consultative specialist, which inspects, converts
   only what the caller declares, and writes nothing
4. `scripts/benchmark_normal_map_io.py` and its report: the round trip, the colour-space mistake,
   the curl bands, 32 handedness readings and the displacement quantization

Measured: the round trip costs 0.08 to 0.25 degrees mean and 0.69 degrees worst with Z
reconstructed; storing the map as sRGB colour and reading it back as data costs 40.9 to 45.0
degrees; the handedness is decided by integrability in 8 of 32 readings with 0 decided wrongly and
an exact tilted plane is genuinely ambiguous; and a height map in a normal slot is named from its
0.442 mean unit-length deviation. See [the gate record](NORMAL_MAP_IO_GATE.md).

The gate corrected two of its own a-priori claims, which is the useful part: the handedness is
*not* invisible to every statistic, and the first two curl estimators were wrong in different ways
(an unpaired edge difference invented curl at a mask border, and central differences were blind to
a slope alternating every row).

P8 remains open beyond this gate: object-space to tangent-space conversion, the UV basis and
mirrored UVs, mipmap-safe filtering, block compression, depth from shading, and any evaluation on
real captures or engine imports.

## Fifteenth Bellium-native wave: P8 sixth gate delivered

1. `bellium/material/resample.py`: three published reductions, a level-of-detail chain with the
   drift of every level, and a declared pixel budget
2. `bellium/material/controlled.py`: `sampled_geometry` and `sampled_height`, the same continuous
   surface at another sampling, so a reduction can be scored against something
3. `bellium/hybrid/normal-mip-chain:v0`: the consultative specialist, statistics first, pixels
   only within the budget
4. `scripts/benchmark_normal_resample.py` and its report: the three forms at two factors, the
   chain, the band-limit counter-example and the detectability of the length defect

Measured: a reduction by two costs 0.1772 degrees of direction on a sphere where the naive form
costs 3.7021 as stored with the identical direction, and slope-space filtering costs 0.4696 and is
therefore worse here; the integral is unchanged by all three. The chain drifts 0.09489 of mean Z
over two levels of a sphere against 0.006 for a cone and 0 for a plane, and the length defect it
leaves is twenty-five times smaller than the check that could catch it. See
[the gate record](NORMAL_RESAMPLE_GATE.md).

P8 remains open beyond this gate: object-space to tangent-space conversion, the UV basis and
mirrored UVs, tile and atlas boundaries under a filter, hardware mip generation, compressed
formats, depth from shading, and any evaluation on real captures or engine imports.

## Sixteenth Bellium-native wave: P8 seventh gate delivered

1. `bellium/material/ambient_occlusion.py`: discrete horizon elevation angle ray-marching,
   projecting cosine-weighted hemispherical occlusion into accessibility [0.0, 1.0]
2. `bellium/specialists/ambient_occlusion.py`: consultative specialist `bellium/hybrid/ambient-occlusion:v0`
   accepting height grids or normal fields
3. `scripts/benchmark_ambient_occlusion.py` and its report on controlled V-grooves, pits and
   standard topographies

## Seventeenth Bellium-native wave: P10 first gate delivered

1. bellium/drafting: Y-up document, mm or in, layers, line/polyline/circle/arc/text
2. SVG drafting profile and DXF R12 ASCII subset with INSUNITS and LIMMAX
3. consultative specialist bellium/deterministic/drafting-document:v0
4. five hand-authored fixtures and scripts/benchmark_drafting.py

The document is the baseline. SVG and DXF must reconstruct it or the specialist
abstains. No neural tier. Offset, cotation and 3D CAD remain later gates. See
the gate record (DRAFTING_GATE.md).
