# P8 gate: reducing a normal field, and the drift of a level-of-detail chain

The gate asks what a reduction does to a normal field. The honest answer separates
three things that are easy to confuse: the direction, the length, and the content
that a coarser grid cannot carry at all.

    python scripts/benchmark_normal_resample.py

Report: benchmarks/manifests/normal-resample-v1.json.

## What exists

| Piece | Behaviour |
| --- | --- |
| downsample_normals | integer-factor reduction in three published forms; a block counts only when every source pixel under it is valid |
| renormalized | averages the unit vectors and normalizes the result: the default |
| naive | stores the averaged vector as it comes out, length defect included |
| slopes | averages the slopes and rebuilds the normal, which is the textbook advice |
| mip_chain | the levels, the mean Z of each one, the coverage, the length the filter produced and the drift of the chain |
| normal-mip-chain v0 | the specialist: builds the chain, reports the drift, and returns pixels only within a declared budget |

## Measured: the direction is not what changes

Every number below compares a reduction of the 32 x 32 controlled field with the
analytic field of the same continuous surface sampled at 16 x 16 (the wave period
scales with the grid; the other families are defined on a normalized domain).

| Field | Factor | Method | Error as stored | Error of the direction | Stored length |
| --- | --- | --- | --- | --- | --- |
| sphere | 2 | renormalized | 0.1772 deg | 0.1772 deg | 1.00000 |
| sphere | 2 | naive | 3.7021 deg | 0.1772 deg | 0.99757 |
| sphere | 2 | slopes | 0.4696 deg | 0.4696 deg | 1.00000 |
| sphere | 4 | renormalized | 0.4961 deg | 0.4961 deg | 1.00000 |
| sphere | 4 | naive | 7.6438 deg | 0.4961 deg | 0.99113 |
| sphere | 4 | slopes | 1.3076 deg | 1.3076 deg | 1.00000 |
| cone | 2 | renormalized | 0.1095 deg | 0.1095 deg | 1.00000 |
| cone | 2 | naive | 2.8159 deg | 0.1095 deg | 0.99835 |
| cone | 2 | slopes | 0.1095 deg | 0.1095 deg | 1.00000 |
| tilted plane | 2 | any | 0.0000 deg | 0.0000 deg | 1.00000 |

Three readings, and the first one is the reason the column exists:

- **naive and renormalized have exactly the same direction.** Averaging and then
  normalizing is the same direction as averaging alone. The 3.70 against 0.18
  degrees is the stored length read through a formula that assumes unit vectors: two
  parallel vectors have no angle, whichever length they carry.
- **the slopes form changes the direction for the worse** on a curved field: 0.4696
  against 0.1772 at a factor of two, 1.3076 against 0.4961 at four. Averaging in
  slope space over-weights the steep pixels, and near the rim of a dome those are
  exactly the pixels where the field turns fastest. On a cone the two forms agree
  exactly, to every digit measured, because a cone's vertical component is constant
  and both estimators then reduce to the same formula.
- **the height column does not separate the methods at all.** A scale on a normal
  cancels in the slope and the integral is scale invariant, so the three forms
  integrate to the same map to six decimals. That column prices the resolution, not
  the filter: 0.192 of relative error at 16 x 16 and 0.337 at 8 x 8 on a sphere.

## Measured: the content a coarser grid cannot carry

The wave family is the counter-example that keeps the table honest. Its period is
declared in pixels, so the benchmark scales it with the grid to keep the surface
identical, and even then a twelve-pixel period is not band limited at sixteen
pixels:

| Wave period | Factor | Error, every method |
| --- | --- | --- |
| 12 px | 2 | 12.85 deg |
| 12 px | 4 | 34.53 deg |
| 24 px | 2 | 6.16 deg |
| 24 px | 4 | 17.81 deg |

All three methods land within 0.9 degrees of each other, and doubling the period
halves the error. What the reduction loses there is frequency, and no filter choice
recovers it: the coarse grid cannot represent it.

## Measured: the chain, and the flattening

Renormalized reduction, thirty-two pixels down twice, against the analytic field of
each level:

| Field | 16 x 16 | 8 x 8 | Mean Z drift | Warnings |
| --- | --- | --- | --- | --- |
| sphere | 0.1772 deg | 0.4982 deg | +0.09489 | flattening_drift, partial_coverage |
| cone | 0.1095 deg | 0.4969 deg | +0.00606 | partial_coverage |
| waves | 12.8502 deg | 34.5281 deg | +0.01198 | none |
| tilted plane | 0.0000 deg | 0.0000 deg | 0.00000 | none |

The drift is the change in mean Z across the chain, and it is the flattening a mip
chain shows: averaging tilts a curved field towards the viewer, by 0.095 of mean Z
over two levels of a sphere, while a cone moves 0.006 and a plane not at all. The
declared limit of 0.03 sits between them, which is the only reason it is 0.03.

The coverage warning appears on the closed silhouettes: a block at the rim of a disc
is never fully covered, so those cells are left invalid rather than filled from half
a neighbourhood.

## Measured: the length defect is invisible to the checks that exist

| Factor | Method | Stored length | Mean unit-length deviation | Inspector verdict |
| --- | --- | --- | --- | --- |
| 2 | naive | 0.997802 | 0.002283 | tangent-space |
| 2 | renormalized | 1.000000 | 0.001460 | tangent-space |
| 4 | naive | 0.990997 | 0.006995 | tangent-space |
| 4 | renormalized | 1.000000 | 0.001883 | tangent-space |

Two caveats, both measured. The inspector's non-unit limit is 0.2, roughly
twenty-five times the defect a chain leaves, so it cannot see this at all. And the
encoding path in this repository normalizes every normal before it writes one, so a
naive chain pushed through encode_normal_map arrives healed; the row above prices a
writer that does not normalize.

## The honest reading

What this gate proves is arithmetic on controlled fields: the direction of a
reduction is the direction of the average, the length is a separate defect that the
integrator and the inspector both ignore, and slope-space filtering measurably
loses the direction on a dome. It prices the resolution loss separately from the
filter loss, and it shows that the flattening a chain produces is visible in the
mean Z before it is visible anywhere else.

What it does not prove: no real texture, no GPU filter and no mipmap generator was
involved. The fields are 32 x 32, the reduction is a box average with an integer
factor, and there is no kernel, no anisotropic footprint and no UV-aware wrap.

Not established: hardware mip generation, sRGB and compressed-format sampling,
anisotropic filtering, tile and atlas boundaries (a block that straddles two atlas
regions), premultiplied alpha, and the shading difference a length defect makes in a
real renderer rather than in a formula that assumes unit vectors.

