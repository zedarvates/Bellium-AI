# Specialists currently implemented locally

Nothing in this list is production-authoritative. Registry presence does not grant
permission to change a live system.

## Imported micro-NNs (legacy-botte)

Eleven tiny classifiers copied from Botte Secrete with SHA-256 provenance.

They load, predict and abstain. They stay in observe mode.

## Native k-NN

- patch-inpaint v0: fill a small selected hole from nearby patches.
- color-cutout v0: isolate a hard-edge object from a simple background.
- asset-quality v0: deterministic licence/integrity gates, then a family-local
  shadow verdict. Private neighbor ledgers are not imported.

- language-phoneme v0: retrieve nearest labelled phones in one language.
- pronunciation-similarity v0: compare a pronunciation to a labelled reference.

- tool-router k-NN v0: choose a tool from a closed catalog of exemplars.
- memory-reranker v0: rank memory titles in one domain, never return bodies.

- visual-anomaly v0: say whether an image still looks like the known states of one domain.

## Native micro-NN

- inpaint-router v0: say whether patch k-NN is enough or the job should escalate.

- tool-router micro-NN v0: propose none, use_tool or escalate.

## Hybrid

- white-background-normalizer v0: cutout then composite on white, or refuse.

- tool-router hybrid v0: hard veto, then k-NN, then micro-NN. Nothing is executed.

## Still missing from the first native wave

The first native wave is now present locally. No specialist is active.

## Second wave

- grapheme-phoneme v0: spelling to sounds and back. Exact attested pairs can be
  certified. Neighbors can only suggest, never certify.

- cognate-retrieval v0: find related word forms in other lects of the same family.
  A labelled set can be linked. Similarity never invents a proto-form.

- prosody-profile v0: compare a syllable rhythm to labelled contours. No audio.
  A match is never certified as a speaker or a reconstructed melody.

- panel-safe-crop v0: crop a panel without cutting required content. The k-NN
  only proposes a margin. Not a comic parser.

- speech-bubble-region v0: find bright compact regions that look like balloons.
  Dialogue is never read or stored. Not a comic parser.

## Third wave

- consistency-retrieval v0: retrieve a labelled visual style in one family.
  Pixels are not changed. Mixed neighbors abstain.

- flat-color-helper v0: fill a hole with the median colour only if the field is
  flat. Textured zones are not flattened.

- voice-activity v0: speech vs silence from compact frame measures. The waveform
  is discarded. Not transcription.

- recording-quality v0: clean, noisy or clipped from compact frame measures. The
  waveform is discarded. Nothing is denoised or repaired.

## Compression research baselines

- byte-compression k-NN v0: causal neighbors predict bytes; zlib stores exact residuals.
- byte-compression micro-NN v0: a single adaptive integer linear neuron predicts
  bytes; decoder repeats online updates. No pretrained model or hidden layers.
- image-compression v0: exact L/RGB/RGBA pixels, including transparent colors.
- stream-compression v0: independent, bounded lossless byte packets.
- splats-compression v0: finite float32 records, static or explicit-base temporal
  residuals. The research format is separate from .fovea.
- ply-archive v0: complete supported binary PLY files, header and every declared
  scalar field included, via reversible byte planes. Values stay opaque, so NaN
  payloads and unknown vendor columns survive. Canonical ASCII files are archived
  too, verbatim or transposed by column when the rows are canonical. Files above
  16 MiB need an explicit `limit`, up to 256 MiB.
  Polygonal files are supported: a `face` element with an index list is archived
  exactly and transposed too when every face has the same arity; mixed arity is
  left verbatim. Multi-element ASCII is supported and always stored verbatim,
  with the column layout reserved for canonical uniform rows. See the coverage,
  polygonal and real-corpus gate records.

All remain consultative. Automatic mode compares actual packet sizes against
classical codecs. See [the model card](model-cards/compression-v0.md) for limits.

## Fourth wave: game assets and NPC behaviour

- texture-repeat-xy v0: measured repeat periods per axis plus a family-local
  verdict. UV repeat counts require a declared target extent. Pixels are never changed.

- texture-tileability v0: wrap continuity per axis. The deterministic gap baseline
  and the neighbours must agree, otherwise the axis abstains.

- nano-tile-seam v0: the first nano tier model, six deterministic measures in and
  a seam verdict out. 38 parameters with a hard 64-parameter budget.

- texture-tile-fixer v0: bounded mirrored feathering across a confirmed wrap plane.
  Returns a candidate with the band, the changed channels and the gap before and
  after. The source image is preserved.

- sprite-anchor v0: proposes an anchor kind (feet, center) and its position inside
  the content box for one sprite family. Never cuts pixels.

- sprite-frame-prep v0: crops to the declared mask, pads, and places the anchor.
  A silhouette that already touches the frame refuses preparation instead of
  being trimmed.

- npc-behavior-router v0 (micro-NN and hybrid): proposes idle, patrol, investigate,
  engage or retreat from caller-declared compact state. Deterministic vetoes run
  first and nothing is executed. The classifier imitates an authored rule.

## Fifth wave: sprite sheets and action consequences

- clip-loop v0: says whether a clip closes its loop from the first and last frame.
  The known clean loops of one family form the reference space; the band between
  that space and a clear drift abstains.

- frame-phase v0: names hold, build, impact or recover for one frame transition.
  No model is shipped: the measurement showed the published rule is not beaten, so
  the rule answers and the result reports baseline_only.

- sprite-sheet-prep v0: measures the grid, splits the sheet into frames, names each
  frame phase and checks the loop. Frames come back as a report with boxes; pixels
  only on request, and nothing is exported.

- consequence-precedent v0: retrieves verified precedents for a proposed action in
  one family. Unverified records never decide.

- consequence-predictor v0 (micro-NN and hybrid): estimates safe, review or
  dangerous. An irreversible action without a dry run or a backup is vetoed before
  any model, and the learned tiers must reproduce the published risk rule.

## Sixth wave: physical estimates (P7 first gate)

- physical-case-retrieval v0: interpolates a physical quantity from a table of
  reference-model cases, by bracketing on one axis or bilinearly on a grid. It
  abstains outside the declared domain, without bracketing cases, on insufficient
  neighbour support and when neighbouring cases disagree.

- gravity-residual v0 (nano-NN): four parameters on a physical basis
  (sin^2 latitude, sin^4 latitude, altitude). Independent benchmark RMSE 2.33e-5 m/s2,
  twenty times more accurate than the 117-case table at twenty-seven times less data.

- atmosphere-model v0 (micro-NN): 21 parameters on the log pressure ratio, worst case
  539 Pa on the benchmark grid against a 1 013 Pa preview threshold.

- physical-estimate v0 (hybrid): reference first, cheaper tiers reported beside it with
  their live deviation, and an explicitly provisional answer when the reference cannot
  run. Units, assumptions, valid range, provenance and uncertainty travel with it.

The reference formula remains the answer whenever it can run: it is exact, smaller and
faster than every alternative measured here.

## Seventh wave: texture grain and the P8 first gate

- texture-orientation v0: names the repeat direction (repeat-x, repeat-y, repeat-both,
  repeat-diagonal, no-repeat) and reports the angle and the period. Directions come from
  the projection method, axes are confirmed by shift differences with bilinear sampling,
  and a repeat that changes across the image abstains instead of becoming one period.

- The repeat specialist gained the same perspective guard, and the period measurement
  moved from a mean projection to shift differences, which is what makes a symmetric
  checkerboard measurable. See [the gate record](TEXTURE_REPEAT_GATE.md): 0 of 18 held-out
  cases invented a period, and rotated grains were named correctly in 4 of 4.

## Eighth wave: animation timing

- easing-profile v0: names the easing curve of a motion series from exemplar retrieval
  over the published curves. The nearest curve must fit, beat the runner-up and agree
  with the published threshold rule, otherwise it abstains.

- animation-timing v0 (hybrid): reports holds, the peak transition, the spacing
  roughness, the easing curve and duplicate frames, with warnings for uneven spacing, a
  peak on the last frame, a missing trailing hold, duplicates and unconfirmed easing.
  Nothing is retimed, trimmed or removed.

- No neural tier was retained here: a 57-parameter candidate reached 0.477 agreement
  against 0.658 for the published rule under noise, so no weights were written.
  See [the gate record](ANIMATION_TIMING_GATE.md).

## Ninth wave: albedo and illumination separation

- albedo-separation v0: splits a render into albedo and illumination, or reports that a
  single image cannot decide. Log-domain separation with a box-blur estimate, anchored
  on a documented percentile, with the identity answer always computed beside it.

- A declared shading prior settles the answer: smooth separates, none returns the render
  as the albedo. Without a prior the specialist returns both candidates, the evidence
  (detail energy, whether the blurred image repeats, shading variation, clipped ratio)
  and flags the choice as unresolved.

- Measured on 50 controlled renders with known maps: identity exact under constant
  illumination, separation winning on structured albedos under smooth shading, and an
  image-only recommendation right in 76 % of cases. See
  [the gate record](MATERIAL_SEPARATION_GATE.md).

## Tenth wave: normals from multi-light captures

- photometric-normals v0: recovers normals and albedo from several captures under
  declared light directions, per pixel least squares with an ambient term, and reports
  per-patch trust from the fit residual. The light vectors are inputs, not measurements.

- Measured on 36 controlled cases: trusted patches at 0.0094 degrees on Lambertian
  captures against 7.86 degrees for the rejected ones, degradation under mild specular,
  and abstention in 10 of 12 strongly specular cases. The blind spot is documented: a
  nearly constant specular term hides in the ambient term.

- No neural tier was retained: a 38-parameter candidate gave 7.64 degrees where the
  published residual rule gives 4.34 degrees at equal coverage, so no weights were
  written. See [the gate record](PHOTOMETRIC_NORMALS_GATE.md).

## Eleventh wave: direct image editing

- direct-filters v0: replays a declared recipe of grayscale, sepia, invert,
  brightness/contrast, saturation, tint and threshold with adjustable strength, on the
  whole image or through a feathered selection mask. The source is never written to.

- magic-eraser v0: proposes a bounded fill for a selected region by reusing the inpaint
  router and patch k-NN. An uncertain selection escalates instead of being filled, and a
  confirmed one returns a preview with certified false.

- No neural tier was retained for the filters: every operation is exact integer
  arithmetic, so a learned tier would add parameters without adding a decision. The
  benchmark measures 4 to 18 ms per 64 x 64 preview and 34 to 163 ms at 192 x 192, and
  states plainly that this is not interactive. See [the gate record](EDITING_GATE.md).

## Twelfth wave: an interactive editing path

- Three interchangeable filter backends: the vectorised path when NumPy is importable,
  256-entry lookup tables for the channel-wise filters otherwise, and the plain per-pixel
  path as the reference. A parametrised test keeps them pixel-identical, and that check
  found three real rounding defects before any of them shipped.

- Previews are area-averaged to a declared side and reported as previews, with scale,
  size and export size, never as exports. A preview session converts the source once and
  then repeats previews at 14.7 ms p95 on a 512 x 512 source, inside the declared 33 ms
  budget, where a cold one-shot preview costs about 0.5 s. See
  [the interactive record](EDITING_INTERACTIVE_GATE.md).

## P4 gate: atlas packing

- atlas-packing v0 (hybrid): places declared frames on bounded pages and returns the geometry,
  with a padding gutter and a page cap. Frames that do not fit are reported unplaced, never
  scaled, rotated or cropped. The plan is re-checked before it is returned, and the two
  published shelf baselines stay in the same module for comparison.
  See [the gate record](ATLAS_PACKING_GATE.md).

- engine-import-validation v0 (hybrid): checks an atlas plan against a declared target contract
  and returns the findings plus a manifest mirror. No engine runs, no file is written, and a
  manifest that fails its own checks abstains instead of shipping. See
  [the gate record](ENGINE_IMPORT_GATE.md).

## P8 gate: relative height from a normal field

- normal-to-height v0 (hybrid): integrates a normal field, from the sibling specialist or from
  anywhere else, into a relative height. Four published integrators are compared in the same
  module and the run reports which one ran, whether it finished, what the declared grazing floor
  refused, and the additive constant that a normal field cannot carry. A field with no recoverable
  slope abstains. See [the gate record](NORMAL_INTEGRATION_GATE.md).

- Measured on four controlled geometries: the row/column average means 0.0682 of relative error
  against 0.1519 for the row alignment, 0.1242 for the vertical alignment and 0.3045 for the
  least-squares solve at 242 ms. On the controlled capture chain the grazing floor takes the error
  from 1.370 to 0.194 while keeping 96 % of the pixels, which is a prior about the capture rather
  than about the integrator.

- integration-method v0 (k-NN): names the integrator to call before anything is integrated, from
  five deterministic features of the slope field, with the published rule returned beside the
  recommendation. The memory holds 16 measured exemplars and the evaluation is held out on other
  seeds: 31 of 32 fields at the best method, against 25 of 32 for the rule and 23 of 32 for the
  fixed default. See [the gate record](NORMAL_INTEGRATION_GATE.md).

- The integration itself keeps no learned tier: once the slopes are known the problem is linear
  algebra with an exact solver per surface family, and the only decision worth learning is which
  solver to call.

## P8 gate: normal maps as pixels

- normal-map-convention v0 (hybrid): reads what a map states about its own encoding, names the
  decidable symptoms (not-a-normal-map, object-space, inverted-z, tangent-space, ambiguous), and
  converts the tangent-space handedness the caller declares. The converted pixels are returned as
  data and no file is written. See [the gate record](NORMAL_MAP_IO_GATE.md).

- The handedness is decided by integrability where the surface bends: negating the green channel
  without mirroring the domain is not the gradient of any surface, so the wrong reading carries a
  curl of twice the cross derivative of the x slope. Measured over 32 readings: 8 decided,
  24 abstained, 0 decided wrongly, and an exact tilted plane is genuinely ambiguous.

- The measurement priced two pipeline mistakes: storing the map as sRGB colour and reading the bytes
  back as data costs 40.9 to 45.0 degrees, and a height map in a normal slot is named from its
  0.442 mean unit-length deviation. Displacement quantization costs 0.196 % of the span at 8 bits.

## P8 gate: reducing a normal field

- normal-mip-chain v0 (hybrid): reduces a normal field into level-of-detail levels in three
  published forms, reports the mean Z, coverage and length of every level, and refuses to
  materialize a chain above a declared pixel budget. See
  [the gate record](NORMAL_RESAMPLE_GATE.md).

- Measured: the direction of a reduction is the direction of the average, so the naive form and the
  renormalized form differ only in the stored length (3.7021 degrees against 0.1772 on a sphere at
  a factor of two, identical 0.1772 of direction). Slope-space filtering measurably loses the
  direction on a dome (0.4696 against 0.1772) and is exactly equal to the renormalized form on a
  cone. The integral cannot see any of it, because a scale on a normal cancels in the slope.

- The chain reports the flattening it produces: mean Z drifts by 0.09489 over two levels of a sphere
  against 0.006 for a cone and 0 for a plane, and the declared limit of 0.03 sits between them.

## P8 gate: ambient occlusion from relief fields

- ambient-occlusion v0 (hybrid): calculates screen-space / heightfield horizon-based ambient
  occlusion accessibility in [0.0, 1.0] from a relative height field or normal map. Evaluates
  cosine-weighted solid-angle visibility along discrete radial directions. See
  [the gate record](AMBIENT_OCCLUSION_GATE.md).

- atlas-raster v0 (hybrid): composes a verified plan into exact pages and returns either pixels
  or 8-bit RGB/RGBA PNG bytes. Pages are re-derived and re-checked first, the encoded files are
  inspected and decoded back, and nothing is written to disk. See
  [the gate record](ATLAS_RASTER_GATE.md).

- atlas-dedup v0 (hybrid): removes duplicate frames, stores one copy per distinct image and returns
  the alias map plus the placements needed to draw every declared frame. Merging is exact unless
  the caller declares a max_delta per-channel bound, in which case the worst measured delta is
  reported, and mirrors are matched under none, flip-x, flip-y or rotate-180 with the transform
  declared per alias; the alias map is re-checked and nothing is written to disk. See
  [the gate record](ATLAS_DEDUP_GATE.md).

## Expérimental

- adaptive-inpaint v1 (experimental): choisit entre copie de patches et interpolation
  entre bords connus. Deux méthodes doivent valider leurs contrôles de contexte.
  Non inscrit au registre, aucune autorité. Voir docs/VISUAL_QUALITY_V3.md.

## ImageMagick replacement suite (P0 visual primitives)

- resample-edge v0 (nano-NN): 38-parameter nano model predicting high-frequency subpixel residuals along detected structural edges during image resizing, converging to exact bilinear interpolation on flat zones (diff_mae = 0.0).
- resample-knn v0 (k-NN): subpixel exemplar gradient matching for oriented edge reconstruction.
- quantize-tone v0 (nano-NN): 25-parameter nano model acting as an adaptive dithering arbiter, suppressing noise artifacts on flat surfaces while modulating Floyd-Steinberg error diffusion across subtle color gradients to prevent posterization.
- palette-match v0 (k-NN): perceptual color space nearest-neighbor mapping with quantization error and coverage diagnostics.
- adaptive-filter v0 (nano-NN): 30-parameter nano model performing local pixel-wise blending between sharpening structural contours (up to 94% on lineart) and smoothing noisy textures.
- filter-selector v0 (k-NN): classifier selecting the optimal convolution preset from 6 statistical image descriptors.
- tone-curve v0 (micro-NN): 108-parameter micro model parameterizing smooth exposure curves (gamma, lift, gain, pivot) from 8 global luminance distribution quantiles.
- adaptive-threshold v0 (k-NN): binarization strategy classifier (Otsu vs adaptive local window) with automatic abstention on flat uniform inputs.
- magick-replacement v0 (hybrid): unified ImageMagick replacement engine exposing bellium-magick CLI and pure-Python API for resizing, quantization, spatial filtering, tone adjustment, thresholding, mathematical morphology, compositing/blending, channel splitting/merging, image comparison (RMSE/PSNR/SSIM) and orthogonal geometry transforms without any C runtime dependency. See [the gate record](MAGICK_REPLACEMENT_GATE.md).
- raster-to-svg v0 (hybrid): traces hard-edge quantized regions into SVG paths. k-NN names flat fills vs thin strokes and abstains on photographic colour counts. The pelican-bicycle fixture is original geometric clip-art for tracing, not a prompt-to-image benchmark.

## P10 gate: 2D drafting document

- drafting-document v0 (deterministic): native Y-up drawing in mm or in, with layers and
  exact line, polyline, circle, arc and text primitives. Emits a drafting SVG profile and
  a DXF R12 subset only after both round-trips reconstruct the document. Foreign SVG,
  splines and empty sheets abstain. Not a pixel tracer and not a CAD kernel. See
  [the gate record](DRAFTING_GATE.md).
