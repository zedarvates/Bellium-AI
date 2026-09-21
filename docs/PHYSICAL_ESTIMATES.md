# Physical estimates: first P7 gate

Units, reference cases and the gravity/weight and pressure baselines, measured
against the cheaper tiers. Every estimate carries its unit, assumptions, valid
range, provenance and declared uncertainty. The published reference remains the
authority; the cheaper tiers exist for runtimes that cannot evaluate it.

## Units and references

Conversions are explicit and dimension-checked, and temperature uses an affine
offset rather than a bare factor. Published references implemented here:

| Reference | Model | Domain |
| --- | --- | --- |
| Normal gravity | WGS-84 Somigliana plus a second-order free-air reduction | latitude -90..90, altitude 0..20 000 m |
| Atmosphere | ISA troposphere (0..11 km) and lower stratosphere (11..20 km) | altitude 0..20 000 m |
| Hydrostatic pressure | incompressible column, gauge pressure | depth >= 0, density > 0 |
| Dynamic pressure | definition only; a drag force still needs a measured coefficient | density > 0, speed >= 0 |
| Ballistic preview | apex height, apex time and the inverse launch speed | no drag, constant gravity |

Spot values used as tests: g(0) = 9.7803253359, g(45) = 9.8062, g(90) = 9.8321849378,
free air about -3.086e-3 m/s2 per kilometre, ISA sea level 288.15 K / 101 325 Pa /
1.225 kg/m3, 11 km 216.65 K / 22 632 Pa, 20 km 5 474.9 Pa. No field measurement was
compared, so the declared uncertainty is the model domain and its approximation
order, never a measured error.

## Task thresholds, declared before training

These are preview thresholds for games and simulations, not metrology targets:

| Family | RMSE | Worst case | Why |
| --- | --- | --- | --- |
| gravity | 1e-3 m/s2 | 5e-3 m/s2 | 0.1 mGal moves a 1.27 m jump apex by about 0.13 mm |
| atmosphere pressure | 1 % of sea level (1 013 Pa) | 3 % (3 040 Pa) | far below the standard atmosphere's own daily variation |

## Measured comparison

The command below measures a grid used neither for training nor for the trainers'
held-out check, and writes benchmarks/manifests/physical-estimates-v1.json:

    python scripts/benchmark_physical_estimates.py

Gravity, 1 200 queries:

| Tier | RMSE (m/s2) | Worst (m/s2) | p50 | Data |
| --- | --- | --- | --- | --- |
| reference | 0 | 0 | 0.0003 ms | none, code only |
| case table (117 cases) | 4.57e-4 | 8.58e-4 | 0.055 ms | 26 887 B |
| nano regressor (4 parameters) | 2.33e-5 | 6.41e-5 | 0.152 ms | 992 B |

Atmosphere pressure, 80 queries:

| Tier | RMSE (Pa) | Worst (Pa) | p50 | Data |
| --- | --- | --- | --- | --- |
| reference | 0 | 0 | 0.0003 ms | none, code only |
| case table (41 cases) | 13.97 | 35.0 | 0.019 ms | 7 847 B |
| micro regressor (21 parameters) | 161.8 | 539.2 | 0.145 ms | 1 183 B |

Reading of the measurement:

- The reference wins outright: exact, smallest and fastest. Nothing here promotes a
  cheaper tier over a formula that is already available.
- On gravity the four-parameter model is twenty times more accurate than the case
  table while being twenty-seven times smaller, because its inputs are the physical
  basis of the relation. A ReLU network of the same budget was one to three orders of
  magnitude worse on the same data: the basis mattered far more than depth.
- On pressure the table is more accurate than the tiny network and six times larger.
  Both stay far inside the preview threshold.

## Abstention

The approximations compare themselves with the reference and abstain outside their
declared band, so an approximation never reports a value it cannot stand behind
while the reference is available. With that live check, none of the 1 280 benchmark
queries abstained. A runtime that cannot evaluate the reference passes
check_reference=False and inherits the published band instead; the result then says
deviation_checked false and stays provisional. Case retrieval abstains outside its
declared domain, without bracketing cases, with insufficient neighbour support, or
when neighbouring cases cannot rebuild each other.

Not established: humidity, contact and coupled-force relations (they need reference
data this gate does not have), terrain or mass anomalies, real weather, drag
coefficients, target-device latency and peak memory.
