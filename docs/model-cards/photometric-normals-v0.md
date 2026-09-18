Model card: bellium/hybrid/photometric-normals:v0

- Task: recover surface normals and albedo from several captures of one surface under
  declared light directions.
- Authority: consultative. pixels_changed stays 0 and the captures are never rewritten.
- Inputs: a list of RGB images, one per declared light direction (at least four), an
  optional gamma, and the light vectors, which are inputs and not measurements.
- Method: per-pixel least squares for albedo times the normal plus an ambient term, then
  the albedo is split back out and a residual is computed against the full model.
- Trust: 8-pixel patches are trusted when their mean residual stays at or below 0.02.
  Trusted, rejected and unset patches are all reported; nothing is smoothed.
- Refusals: unknown light directions are refused outright; fewer than four lights raise;
  captures too small for one patch abstain with captures_too_small; a trusted coverage
  below 10 % or a mean residual above 0.1 abstains with no_reliable_patch or
  lambertian_mismatch.
- Measured on 36 controlled cases: trusted patches at 0.0094 degrees on Lambertian
  captures against 7.86 degrees for the rejected ones, degradation under mild specular,
  and abstention in 10 of 12 strongly specular cases. See the gate record.
- Blind spot: a nearly constant specular term is absorbed by the ambient term, so a
  planar specular capture stays fully trusted with 15 to 21 degrees of error.
- Limits: controlled synthetic captures, unambiguous light directions, Lambertian
  surfaces, no interreflection or coloured lights, no roughness or height output, and
  about 30 to 48 ms per 32 x 32 case.
- Evidence: tests/test_photometric_normals.py, scripts/benchmark_photometric_normals.py.
