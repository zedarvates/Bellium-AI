Model card: bellium/micro-nn/atmosphere-model:v0

- Task: approximate ISA pressure for constrained runtimes.
- Authority: consultative, and provisional when it answers alone.
- Inputs and target: normalised altitude, its square and its cube; the target is the
  logarithm of the pressure ratio p/p0 divided by 3, so the network fits the nearly
  linear log-pressure instead of the exponential ratio.
- Architecture: layers [3, 4, 1], 21 parameters, 1 183 bytes on disk.
- Training: 201 samples from 0 to 20 000 m every 100 m, seed 43, momentum 0.9
  (scripts/train_atmosphere_regressor.py). Held-out every 100 m at a 50 m offset:
  RMSE 0.269 % of sea level, worst case 0.538 % (545 Pa).
- Independent benchmark grid (25 m offset): RMSE 161.8 Pa, worst case 539.2 Pa, against
  a 1 013 Pa task threshold.
- Reference: bellium/reference/isa. The result reports the reference value, the live
  deviation, the declared band and the published worst case.
- Abstention: with check_reference true the model abstains outside its declared band
  (1 013 Pa). With check_reference false the deviation is unverified and reported as such.
- Limits: standard atmosphere only. No weather, no humidity, no wind, no terrain. The
  case table of the same family is six times larger and about twelve times more accurate;
  both stay far inside the preview threshold.
- Evidence: tests/test_physical_specialist.py, scripts/benchmark_physical_estimates.py.
