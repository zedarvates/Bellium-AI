Model card: bellium/hybrid/albedo-separation:v0

- Task: split a render into albedo and illumination, or report that the image alone
  cannot say which low-frequency variation is which.
- Authority: consultative. pixels_changed stays 0: the input image is never rewritten
  and every map is returned as linear floats.
- Inputs: an RGB image, an optional gamma to linearise, a separation radius and a
  declared shading prior (smooth, none or unknown).
- Method: log-domain homomorphic separation with a box-blur illumination estimate,
  anchored on a documented percentile because the absolute illumination level is not
  identifiable from a single image.
- Priors: smooth separates, none returns the render as the albedo, unknown reports the
  documented recommendation, both candidates and the evidence, and flags the choice as
  unresolved.
- Evidence: detail energy, whether the blurred image repeats, shading variation and the
  clipped ratio. A repeating blur means the low frequencies are albedo, not shading.
- Warnings: highlights_clipped (abstains above 2 % clipped), flat_albedo_not_identifiable,
  shading_prior_required.
- Measured on 50 controlled renders: identity is exact under constant illumination
  (0.0000), separation wins on structured albedos under smooth shading, the image-only
  recommendation is right in 76 % of cases, and the failure modes are high-frequency
  shading and flat albedos. See the gate record.
- Limits: controlled synthetic renders only, no photograph, no specular or
  interreflection handling, no normal or roughness map, and about 19 ms per separation
  at 32 x 32, so it is a build-time tool.
- Evidence: tests/test_material_separation.py, scripts/benchmark_material_separation.py.
