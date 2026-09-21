Model card: bellium/knn/physical-case-retrieval:v0

- Task: interpolate a physical quantity from a table of reference-model cases.
- Authority: consultative. A case is a model output, never a measurement.
- Families: gravity (latitude, altitude), atmosphere-pressure and atmosphere-density
  (altitude), hydrostatic (depth, density).
- Interpolation: one-axis families use the bracketing pair; grid families use bilinear
  interpolation on the bracketing rectangle; other tables fall back to inverse-distance
  weighting with a leave-one-out consistency check.
- Memory: models/knn/physics/*.json, rebuilt by scripts/build_physical_cases.py
  (117 gravity cases, 41 pressure, 41 density, 63 hydrostatic).
- Refusals: out_of_domain, no_bracketing_cases, insufficient_neighbor_support,
  neighbors_inconsistent, plus strict input validation with no zero coercion.
- Measured: gravity RMSE 4.57e-4 m/s2 and worst case 8.58e-4 (1 200 queries);
  pressure RMSE 13.97 Pa and worst case 35.0 Pa (80 queries).
- Limits: accuracy follows the table spacing, so a fine grid costs memory; the
  interpolation error is measured by the benchmark against the reference, not
  estimated by the specialist.
- Evidence: tests/test_physical_cases.py.
