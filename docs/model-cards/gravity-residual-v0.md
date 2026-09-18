Model card: bellium/nano-nn/gravity-residual:v0

- Task: approximate local gravity for runtimes that cannot evaluate WGS-84.
- Authority: consultative, and explicitly provisional when it answers alone.
- Inputs and basis: sin^2(latitude), sin^4(latitude) and normalised altitude. The
  physically motivated basis is what makes a four-parameter model work.
- Architecture: layers [3, 1], linear output, 4 parameters, target (g - g0) / 0.05 with
  g0 = 9.80665 m/s2, so the model stores the deviation from standard gravity.
- Budget: 64 parameters and 8192 bytes; measured 4 parameters and 992 bytes on disk.
- Training: 651 samples on a latitude/altitude grid (scripts/train_gravity_regressor.py),
  seed 41. Held-out 1 200 samples on offset grids: RMSE 2.33e-5 m/s2, worst case 6.02e-5.
  Independent benchmark grid: RMSE 2.33e-5, worst case 6.41e-5.
- Reference: bellium/reference/wgs84-somigliana. The result reports the reference value,
  the live deviation, the declared band and the published worst case.
- Abstention: with check_reference true the model abstains outside its declared band
  (1e-3 m/s2, the preview tolerance). With check_reference false the deviation cannot be
  verified and the result says so.
- Comparison recorded: a ReLU network with 21 parameters reached 2.6e-2 m/s2 on the same
  data, three orders of magnitude worse than this four-parameter linear model.
- Limits: normal gravity plus free air only. No terrain, no mass anomalies, no rotation
  or tide terms, and no field measurement was compared.
- Evidence: tests/test_physical_specialist.py, scripts/benchmark_physical_estimates.py.
