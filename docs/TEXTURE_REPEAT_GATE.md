# P8 first gate: repeat, tileability and grain on distorted textures

The gate asks whether repeat detection survives rotated and perspective-distorted
textures, not only clean axis-aligned ones. The measurement runs on held-out
synthetic cases whose parameters appear neither in the shipped memories nor in
their builders.

    python scripts/benchmark_texture_repeat.py

Report: benchmarks/manifests/texture-repeat-v1.json. An angle is accepted within
7 degrees; nothing is promoted by the result.

## Measured result

| Class | Cases | Period found | Period exact | Period invented | Grain correct | Worst angle error | Perspective refused |
| --- | --- | --- | --- | --- | --- | --- | --- |
| periodic | 6 | 3 | 3 | 0 | 6/6 | 0.0 deg | - |
| nonperiodic | 5 | 0 | - | 0 | 5/5 | - | - |
| rotated | 4 | 0 | - | 0 | 4/4 | 5.0 deg | - |
| perspective | 3 | 0 | - | 0 | - | - | 1 of 3, 2 reported no repeat |

Reading of the measurement:

- No held-out case invented a period: 0 of 18. The perspective guard added during
  this gate is what removed the two false global periods that the repeat specialist
  produced before the fix, when a warped texture was reported as one clean period.
- Rotated grains are named correctly in all four cases, with an angle error of at
  most 5 degrees, which is the resolution of the 15-degree coarse scan followed by
  a 2-degree refinement.
- The three tiled textures with tile sizes outside the shipped memory range
  (12, 20, 24 pixels) are not confirmed by the repeat specialist: it abstains for
  lack of comparable cases. The orientation specialist still names them
  repeat-both, so coverage of the repeat memory is the limiting factor, not the
  measurement.
- A mild perspective warp (0.20) is reported as no repeat rather than as a clean
  period. The detection limit is documented rather than hidden.

## What changed during the gate

| Change | Reason |
| --- | --- |
| Shift-difference period measurement with bilinear sampling | A mean projection is blind to a symmetric checkerboard and biases oblique angles |
| Perspective guard in the repeat specialist | Measurement showed invented global periods on warped textures |
| Grain classification with axis-first rule | A checkerboard is also periodic on the diagonal and must not be called a diagonal grain |
| Orientation memory with an authored-versus-measured check | A fixture whose label the measurement contradicts is refused instead of stored |

Not established: real game assets, anisotropic filtering, mip behaviour, UV
unwrapping, and any claim about how a texture looks after tiling. Repeat periods
are pixel counts on the tested images.
