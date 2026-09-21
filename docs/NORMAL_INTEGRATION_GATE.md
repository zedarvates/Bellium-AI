# P8 gate: a normal field becomes a relative height

The gate asks how much geometry a normal field actually carries. The honest answer is: the
gradient, and therefore the height up to one additive constant, and only where the surface is
not seen edge-on. This record states what was measured, which two defects the measurement
found, and what remains unproven.

    python scripts/benchmark_normal_integration.py

Report: benchmarks/manifests/normal-integration-v1.json.

## What exists

| Piece | Behaviour |
| --- | --- |
| gradients | slopes and a validity mask; a normal that lies in the image plane is refused, and min_cosine declares how close is too close |
| integrate_cumulative | integrates along rows and aligns them on the difference of the row integrals |
| integrate_cumulative_vertical | aligns rows on the mean column slope instead |
| integrate_cumulative_average | averages the row-wise and column-wise integrals |
| integrate_least_squares | Gauss-Seidel Poisson solve, fixed sweep count in a fixed order, with the linear part restored from the mean slopes |
| integrate_height | one call: slopes, integration, pixel scale, offset removal, convergence report |
| height_error | RMSE after removing the best offset, plus the error relative to the span of the reference |
| recover_height | the consultative specialist bellium/hybrid/normal-to-height:v0 |
| integration_method | five deterministic features of the slope field, a measured memory of 16 exemplars and the published rule returned beside the recommendation |
| integrability | the cell curl of the slope field, its bands and what no integrator can recover; measured in [the normal-map gate](NORMAL_MAP_IO_GATE.md) |

The offset is not a defect to be hidden: a normal field cannot carry it, so it is removed and
reported separately. The pixel-to-world scale is a declared input as well, because orientation
does not carry size.

## Measured: four integrators on four controlled geometries

32 x 32 fields, least-squares at 400 sweeps, every error with the offset removed and divided by
the span of the reference surface:

| Geometry | least-squares | cumulative | cumulative-average | cumulative-vertical |
| --- | --- | --- | --- | --- |
| sphere | 0.806 | 0.177 | **0.059** | 0.230 |
| cone | 0.158 | 0.159 | **0.025** | 0.178 |
| waves | 0.254 | 0.164 | **0.080** | 0.088 |
| tilted plane | **0.000** | 0.108 | 0.109 | **0.000** |
| mean | 0.3045 | 0.1519 | **0.0682** | 0.1242 |
| p50 latency | 242.2 ms | 1.20 ms | 1.17 ms | 1.14 ms |

The row/column average is the default because it has both the lowest mean and the lowest worst
case. Reading the rest of the table honestly:

- least-squares is the only method that is exact on a constant slope, and it pays for that with
  time: 242 ms against 1.2 ms. It is also the worst on a dome, where 400 sweeps leave the
  large-scale component untouched (last change 0.33, reported as not converged) and the
  remaining error is 0.806. Gauss-Seidel removes detail-scale error quickly and smooth error
  slowly, so the sweep count is a budget and the report says so per call.
- the vertical alignment is exact on a constant slope as well, at 1.1 ms instead of 242 ms, and
  it is the better of the two cumulative forms on an oscillating surface. On a closed silhouette
  the row-difference alignment wins instead. Publishing both numbers is cheaper than deciding
  for the caller.

## Two defects the measurement found

- the cone fixture was a paraboloid: its normals were (x, y, 1) where a cone of z = 1 - r needs
  (x/r, y/r, 1). The fixture was wrong, not the integrator, so the fixture was fixed.
- the row-alignment form drops the vertical slope of a tilted plane, because identical rows carry
  no vertical signal for an alignment that compares row integrals: 0.108 of relative error on a
  surface whose height is exactly linear. That is what produced the vertical alignment above, and
  the comment that claimed exactness was corrected to match the measurement.

## Measured: where the field comes from

A height is only as good as the normals it integrates. The controlled chain runs the published
pipeline: six declared lights, a known albedo, captures, then the normals specialist, then the
integrator. 648 pixels are offered.

| Grazing floor | Pixels kept | Coverage | Relative error, all | Relative error, kept |
| --- | --- | --- | --- | --- |
| 0.0 | 648 | 1.000 | 1.370 | 1.370 |
| 0.05 | 632 | 0.975 | 0.232 | 0.229 |
| **0.1** (specialist default) | 622 | 0.960 | 0.194 | 0.182 |
| 0.2 | 590 | 0.910 | 0.166 | 0.138 |
| 0.3 | 548 | 0.846 | 0.172 | 0.128 |
| 0.5 | 472 | 0.728 | 0.210 | 0.127 |

The same geometry integrated from its own deterministic normals reaches 0.059. The difference is
the capture, not the integrator: near the silhouette the recovered vertical component is not only
small, it is sometimes of the wrong sign, and a slope divided by it is noise. Refusing those
pixels is a declared prior, so the specialist declares it, counts what it dropped, and lets the
caller set the floor to zero. The measured knee is between 0.1 and 0.2; the default stays at 0.1,
which keeps 96 % of the offered pixels.

## The honest reading

What this gate proves is internal and arithmetic: on controlled fields the integrators behave as
the table says, the offset is genuinely unrecoverable and correctly removed, the declared scale
scales the result linearly, and the same input always gives the same output. It also shows a
real cost: a capture-derived normal field needs a grazing floor to be usable, and the best of the
four integrators still leaves roughly 6 % of the surface span on a dome.

What it does not prove: no real photograph, scan or engine asset was integrated, no metric height
was produced, and no comparison against a production integrator from another codebase was run.
The pixel scale is declared by the caller, never measured here.

Not established: depth from shading, occlusion and interreflection, non-integrable fields (the
curl of the recovered field is not measured), multi-view fusion, normal-map convention conversion
(DirectX against OpenGL), displacement export, and any use of the height as geometry rather than
as a map.

## Measured: which integrator to call

The integration itself has no learned tier and does not need one: once the slopes are known the
problem is linear algebra with an exact solver per surface family. The remaining decision is
*which* solver to call, and that one is worth a memory, because getting it wrong costs a factor of
thirteen on a dome (0.806 against 0.059) or two hundred milliseconds on a ramp.

bellium/knn/integration-method:v0 reads five deterministic features of the slope field, retrieves
the closest measured exemplars and names a method. The published rule answers beside it and the
specialist returns both, so a caller can ignore the recommendation at no cost.

The memory holds 16 exemplars: the four controlled surfaces at four declared noise levels
(sigma 0, 0.02, 0.05, 0.1), each labelled with the method that measured best, with all four
relative errors stored beside the label. Ties are broken by the measured p50 latency, so an exact
tie recommends the cheaper method: an exact ramp is labelled cumulative-vertical at 1.14 ms rather
than least-squares at 242 ms.

The evaluation uses seeds 2 and 3, which are not in the memory, so the numbers below are held out:

| Choice | Fields at the best method | Mean relative error |
| --- | --- | --- |
| k-NN recommendation | 31 / 32 (96.9 %) | 0.0635 |
| published deterministic rule | 25 / 32 (78.1 %) | 0.0764 |
| fixed default, cumulative-average | 23 / 32 (71.9 %) | 0.0897 |
| oracle, best method per field | 32 / 32 | 0.0634 |

The recommendation costs nothing measurable against the oracle, and its single miss is a
negligible one: an oscillation at noise 0.1 where the two candidates measure 0.2749 and 0.2728.
What it buys is the ramp. On a perturbed plane the rule answers cumulative-vertical, which
measures 0.0057, where least-squares measures 0.0038, because a rule cannot see noise in an
otherwise constant slope. The disagreement is reported as
learned_tier_overrides_the_published_rule rather than applied silently.

The limit is the fixture set, and it is a real one: four geometries, one noise model, 32 x 32
fields, and a label universe of two methods in the memory against three in scoring, because
cumulative and cumulative-vertical never won on a perturbed field here. This memory documents a
convention on controlled data; it is not evidence about real normal maps.
