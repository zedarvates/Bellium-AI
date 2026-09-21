# P8 gate: a normal map as pixels, and what it can say about itself

The gate asks what a map states about itself: its encoding, the defects that are
decidable from the pixels alone, and whether the tangent-space handedness is one of
them. Two of the three answers were not what this file first asserted, and the
measurement is what settled it.

    python scripts/benchmark_normal_map_io.py

Report: benchmarks/manifests/normal-map-io-v1.json.

## What exists

| Piece | Behaviour |
| --- | --- |
| encode_normal_map | 8-bit RGB, n = 2c - 1, no transfer function, every normal normalized before it is stored |
| decode_normal_map | unit normals back, with a mask that keeps the flat pixels out of the statistics, and Z either reconstructed from the first two channels or read from the third |
| flip_normal_convention | the green channel flips, because that is the one the handedness moves |
| detect_convention | decides the handedness from integrability, or states which of two reasons stopped it |
| inspect_normal_map | the decidable symptoms: not-a-normal-map, object-space, inverted-z, tangent-space, ambiguous |
| integrability | the cell curl of the slope field, in the integration module, with its bands |
| quantize_height | displacement levels plus the span, step and worst error, no file written |
| normal-map-convention v0 | the consultative specialist: inspect, and convert only what is declared |

## Measured: the round trip

8 bits, both ways, over the four controlled fields:

| Field | Z reconstructed, mean | Z reconstructed, worst | Z stored, mean |
| --- | --- | --- | --- |
| sphere | 0.2419 deg | 0.6914 deg | 0.1695 deg |
| cone | 0.1987 deg | 0.3734 deg | 0.1626 deg |
| waves | 0.2489 deg | 0.3178 deg | 0.2413 deg |
| tilted plane | 0.0843 deg | 0.0843 deg | 0.0895 deg |

With a 0.05 perturbation on the normals the reconstructed worst case reaches
3.79 degrees on the sphere, at the pixels where x^2 + y^2 approaches 1 and Z is
reconstructed from almost nothing. That is the honest cost of a two-channel map
near a silhouette, not a defect of the encoder.

## Measured: what treating the map as colour costs

The mapping is n = 2c - 1 with no transfer function. Applying the sRGB curve on the
way in and reading the bytes back as data, which is the pipeline mistake this row
prices:

| Field | Mean error | Worst error |
| --- | --- | --- |
| sphere | 40.93 deg | 68.98 deg |
| cone | 42.98 deg | 51.05 deg |
| waves | 41.79 deg | 43.51 deg |
| tilted plane | 45.04 deg | 45.04 deg |

Forty degrees is not a subtle defect, and no amount of checking the vector lengths
would catch it: the field stays unit length and simply describes a different surface.

## Measured: integrability, and the bands that follow

The curl of the slope field is what no integrator can turn into a surface. It is
taken around a cell of four pixels, and the cell counts only when all four corners
are valid.

| Case | Curl ratio | Verdict |
| --- | --- | --- |
| analytic tilted plane | 0.000 | integrable |
| analytic cone | 0.011 | integrable |
| analytic sphere | 0.069 | integrable |
| analytic waves | 0.125 | integrable |
| a bounded rotation, any strength | 0.168 to 0.246 | suspect |
| a slope that alternates every row | 2.000 | not_integrable |
| waves with a 0.1 perturbation | 0.725 | not_integrable |

The bands are 0.15 and 0.25, and the table says why they are wide. Sampled analytic
normals are not exactly the gradient of the sampled height, so sampling alone costs
0.069 on a sphere and 0.125 on a wave, both of which integrate at 0.06 and 0.08 of
relative error. A bounded rotation saturates near 0.17 whatever its strength,
because the ratio divides by a slope magnitude that grows with the rotation. The
metric is therefore a band, not a threshold, and it is not a prediction of an
integrator's error: the perturbed plane measures 0.44 of curl and integrates at
0.11.

## Measured: the handedness, where this file was wrong first

The first version of this module asserted that the handedness is not inferable from
a map, on the reasoning that flipping the green channel mirrors the surface. The
measurement contradicted it. Negating the green channel without mirroring the
domain is not the gradient of any height: the flipped reading carries a curl of
twice the cross derivative of the x slope. Where the surface bends, the map decides
itself.

Both handedness of the same controlled field were encoded and offered to the
detector, at four noise levels: 32 readings in total.

| Outcome | Count |
| --- | --- |
| decided | 8 |
| abstained | 24 |
| decided wrongly | 0 |

The decisive cases are the clean ones: the sphere separates by 3.18, the waves by
5.13, the sphere at 0.02 by 2.09. The abstentions are the honest half of the result.
An exact tilted plane is genuinely ambiguous, because both readings are integrable
(both readings measure 0.000) and both are real surfaces: the two height maps differ
by the sign of the y slope, and the integrator reaches 0.109 and 0.188 of relative
error on them. A cone at rest separates by 6 times, but the wrong reading still
measures 0.088, below the sampling floor, so the detector stays silent rather than
claim a decision it cannot support.

The detector reads only the two channels that carry the slopes, so an inverted blue
channel does not change its answer, and it needs no pixel scale because it compares
two curls of the same field.

## Measured: the symptoms a map can state

One controlled field per variant, read with its own mask:

| Variant | Verdict | Mean Z | Mean unit-length deviation |
| --- | --- | --- | --- |
| tangent-space | tangent-space | 0.671 | 0.002 |
| flipped, declared as flipped | tangent-space | 0.671 | 0.002 |
| flipped, read as +Y | tangent-space | 0.671 | 0.002 |
| inverted Z | inverted-z | -0.671 | 0.002 |
| object-space | object-space | 0.000 | 0.002 |
| a height map stored in RGB | not-a-normal-map | 0.394 | 0.442 |

The first three rows are identical to every digit measured, which is exactly the
point: the handedness is invisible to the statistics and visible only to
integrability. The last two rows are the defects a caller actually meets, and both
are named correctly.

## Measured: displacement

| Field | Depth | Step | Worst error | Of the span |
| --- | --- | --- | --- | --- |
| sphere | 8-bit | 0.00399106 | 0.00199288 | 0.196 % |
| sphere | 16-bit | 0.00001553 | 0.00000765 | 0.00075 % |
| waves | 8-bit | 0.00521387 | 0.00260694 | 0.196 % |
| waves | 16-bit | 0.00002029 | 0.00001014 | 0.00076 % |

Heights are relative, so the span is the declared or measured range of the valid
pixels, and the report carries it. No file is written: the shipped PNG writer is
8-bit RGB and RGBA only, so a depth map leaves this gate as data.

## Three defects the measurement found

- decoding without a mask counted the pixels that encode a missing normal, which
  carry Z = 1, and read a mean Z of 0.792 where the masked field measures 0.671.
- the first curl estimator used forward differences with an unpaired edge, which
  invented curl at the border of a mask and reported 0.12 on a smooth sphere.
- the second one used central differences over two pixels and was blind to a slope
  that alternates sign every row: the classic odd-even decoupling, measured 0.000
  where the cell form measures 2.000.

## The honest reading

What this gate proves is arithmetic on controlled fields: the encoding round trip
costs a fifth of a degree, the handedness is decided by integrability where the
surface bends and refused where it does not, and four encoding defects are named
correctly from the pixels. It also prices two real pipeline mistakes, the sRGB
transfer at about forty degrees and a height map in a normal slot.

What it does not prove: no real asset was read. Every field here is 32 x 32, from
the same four geometries, and the noise model is one Gaussian applied to normals.
Resolution and noise both work against the detector, and neither was varied, so the
8 of 32 decision rate is a property of this fixture set rather than of normal maps
in general: a real map has sharper spatial gradients than the sampling floor and its
noise is not this Gaussian.

Not established: object-space to tangent-space conversion, the UV basis and mirrored
UVs, mipmap-safe filtering, block compression (BC5 and its siblings), the 16-bit or
PNG path for displacement, and any comparison against a shipped baking tool.

