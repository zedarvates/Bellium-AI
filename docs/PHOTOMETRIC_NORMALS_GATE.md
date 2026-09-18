# P8 third gate: normals from controlled multi-light captures

The gate asks whether surface normals can be recovered from several captures of
the same surface under declared light directions. The captures are generated here
with known normals, which the roadmap allows ("controlled renders with known
maps"); no photograph was used.

    python scripts/benchmark_photometric_normals.py

Report: benchmarks/manifests/photometric-normals-v1.json. 36 cases: 4 geometries
x 3 albedos x 3 specular levels, 32 x 32 pixels, 6 declared lights, 8-pixel patches.

## Measured result

Angular error in degrees. "Trusted" and "rejected" are the patches the residual
rule accepts and refuses.

| Specular | Cases | All valid pixels | Trusted patches | Rejected patches | Trusted fraction | Abstentions | p50 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.00 | 12 | 2.3807 | **0.0094** | 7.8586 | 76.6 % | 0 of 12 | 30 ms |
| 0.10 | 12 | 7.0481 | 5.4801 | 10.4523 | 76.6 % | 0 of 12 | 35 ms |
| 0.25 | 12 | 13.3484 | 17.4419 | 12.5297 | 20.4 % | 10 of 12 | 33 ms |

Reading of the measurement:

- On Lambertian captures the fit is exact where the surface is lit: a plane and a
  wavy surface come back at 0.00 degrees, and the trusted patches of the whole set
  average 0.0094 degrees while the rejected ones carry 7.86 degrees. The gate is
  what separates them, and this is the measured benefit of the residual rule.
- A sphere is the interesting case: its overall error is 9.02 degrees because the
  pixels facing away from every light cannot be fitted, and the gate trusts the
  remaining 40 % of its area at 0.038 degrees.
- Mild specular degrades the trusted set to 5.48 degrees without raising the
  residual enough to refuse it. That is reported, not hidden.
- Strong specular collapses the gate: 0 trusted patches in 10 of the 12 cases, so
  the specialist abstains. The two exceptions are the tilted plane, where a nearly
  constant specular term is absorbed by the ambient term, leaving 100 % of the area
  trusted at 15 to 21 degrees of error. This is the documented blind spot: a
  uniform specular offset is invisible to a residual test.
- Latency is about 30 to 48 ms per 32 x 32 case including capture generation, so
  this is a build-time tool.

## Neural candidate: measured and not shipped

A 38-parameter network was trained to predict patch trustworthiness from six
features. At equal coverage on held-out cases the published residual rule gives
4.34 degrees of mean error where the candidate gives 7.64 degrees, so no weights
were written. This is the third tier measured and deliberately not shipped, after
the animation phase and the easing curve.

Not established: real captures, unknown or uncalibrated lights, interreflection,
non-Lambertian materials, coloured lights, roughness or height maps, and any
integration of normals into a height field.
