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
