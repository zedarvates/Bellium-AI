Model card: bellium/hybrid/normal-to-height:v0

- Task: integrate a normal field into a relative height map, and report which integrator ran, what it
  refused and how far it got.
- Authority: consultative. certified and pixels_changed stay false and 0: nothing is written back and
  no map is promoted to geometry.
- Inputs: normals (3-component numbers or null), optional mask, method, iterations, tolerance,
  pixel_scale and min_cosine. Undocumented query keys are refused rather than ignored.
- Output: the height grid, the validity grid, the number of offered and kept pixels, the offset that
  was removed, the relief in the declared unit, the convergence report and the warnings.
- Methods: least-squares (Gauss-Seidel Poisson solve with the linear part restored from the mean
  slopes, 400 sweeps by default), cumulative (rows aligned on the difference of the row integrals),
  cumulative-average (row and column integrals averaged, the default), cumulative-vertical (rows
  aligned on the mean column slope).
- Measured on four controlled 32 x 32 geometries (relative error, offset removed): cumulative-average
  means 0.0682 with a worst case of 0.109; cumulative 0.1519; cumulative-vertical 0.1242; least-squares
  0.3045 against 242 ms per field. least-squares and cumulative-vertical are both exact (0.000) on a
  constant slope. See benchmarks/manifests/normal-integration-v1.json.
- Convergence: the sweep count is a budget, not a promise. Gauss-Seidel removes detail-scale error
  quickly and smooth error slowly, so every least-squares call reports last_change, the tolerance it
  was compared against and a converged flag; an unfinished solve is returned with
  least_squares_not_converged and a confidence capped at 0.4 rather than presented as a result.
- Grazing floor: a normal close to the image plane gives a slope divided by a near-zero vertical
  component. min_cosine declares the floor and defaults to 0.1 in this specialist, where the controlled
  capture chain measures 1.370 of relative error with no floor against 0.194 with one, keeping 96 % of
  the pixels. What the floor drops is counted in grazing_dropped, never silently smoothed.
- Verdicts: ready, or abstain with no_recoverable_slope when every normal is missing or lies in the
  image plane. Warnings: few_valid_pixels, grazing_pixels_dropped, least_squares_not_converged,
  flat_field.
- Limits: heights are relative. An additive constant is not recoverable from a normal field and the
  pixel-to-world scale is the declared pixel_scale, never inferred. No real capture, scan or engine
  asset was integrated, no metric height was produced, and the curl of the field is not measured.
- Evidence: tests/test_normal_integration.py, scripts/benchmark_normal_integration.py,
  docs/NORMAL_INTEGRATION_GATE.md.

