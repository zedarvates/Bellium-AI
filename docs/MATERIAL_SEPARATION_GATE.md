# P8 second gate: albedo and illumination on controlled renders

The gate asks whether a render can be split into albedo and illumination. The
renders are generated here with their ground-truth maps, which the roadmap allows
("controlled renders with known maps"); no photograph was used and nothing about
real captures is established.

    python scripts/benchmark_material_separation.py

Report: benchmarks/manifests/material-separation-v1.json. 50 cases: 5 albedo kinds
x 5 illumination kinds x 2 seeds, 32 x 32 pixels, separation radius 6.

## Measured result

Mean albedo RMSE, lower is better. "Declared" applies the caller's prior: none for
constant illumination, smooth otherwise.

| Illumination | Identity | Separation | Declared prior | Automatic | Separation wins | Recommendation hits |
| --- | --- | --- | --- | --- | --- | --- |
| constant | 0.0000 | 0.2920 | **0.0000** | 0.0623 | 0 of 10 | 6 of 10 |
| linear | 0.2202 | 0.2814 | 0.2814 | **0.1953** | 6 of 10 | 8 of 10 |
| vignette | 0.2274 | 0.2655 | 0.2655 | **0.1647** | 6 of 10 | 10 of 10 |
| lamp | 0.3158 | 0.1936 | 0.1936 | **0.1767** | 6 of 10 | 10 of 10 |
| shadow-band | 0.2022 | 0.2727 | 0.2727 | 0.2326 | 4 of 10 | 4 of 10 |

Reading of the measurement:

- With constant illumination the render *is* the albedo and the identity answer is
  exact. Frequency separation then does harm (0.2920) because the blur removes the
  albedo's own low frequencies: this is why the prior matters.
- Separation wins whenever the albedo is structured and the shading varies smoothly
  (checker, stripes, color blocks under linear, vignette and lamp): 6 of 10 cases per
  class, and the recommendation finds those 10 of 10 under vignette and lamp.
- A shading band is high-frequency illumination and defeats the method (4 of 10);
  a flat or gradient albedo defeats it too, because then there is no detail left to
  keep. Those are the two documented failure modes.
- The image-only recommendation is right in 38 of 50 cases (76 %). That number is the
  honest measure of how far a single image can go, which is why the specialist asks
  for a declared prior and returns both candidates when it has none.
- Latency is about 19 ms per separation at 32 x 32 with radius 6: a build-time tool,
  not a real-time one.

## What the specialist guarantees

| Guarantee | How |
| --- | --- |
| Both explanations are always returned | `candidates` carries the identity and the separated albedo with their illumination |
| The ambiguity is never hidden | `unresolved`, `shading_prior_required` and the evidence block |
| The absolute level is not claimed | the result is anchored on the 98th percentile and errors are reported absolute and scale-free |
| Clipped highlights do not become a material | abstention above 2 % clipped pixels |

Not established: real photographs, coloured cast removal, specular highlights,
interreflection, normal or roughness maps, and any artistic judgement about the
recovered albedo.
