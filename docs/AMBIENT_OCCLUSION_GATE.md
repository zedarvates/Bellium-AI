# P8 gate: horizon-based ambient occlusion from relief fields

The gate measures screen-space / heightfield ambient occlusion (accessibility)
from continuous surface topographies and normal-derived height fields.

    python scripts/benchmark_ambient_occlusion.py

Report: benchmarks/manifests/ambient-occlusion-v1.json.

## What exists

| Piece | Behaviour |
| --- | --- |
| horizon_ambient_occlusion | discrete radial horizon ray-marching, elevation angle projection, cosine weighting |
| estimate_ambient_occlusion | consultative specialist: accepts height map or normal field, returns accessibility grid |

## Measured

Benchmarked on 32 x 32 test fields with radius 6 and 8 discrete directions:

| Topography | Variant / Parameter | Mean Accessibility | Center Pixel AO | Latency |
| --- | --- | --- | --- | --- |
| V-groove | slope = 0.5 | 0.8350 | 0.536 | ~17 ms |
| V-groove | slope = 1.0 | 0.7367 | 0.470 | ~19 ms |
| V-groove | slope = 2.0 | 0.6612 | 0.421 | ~20 ms |
| Hemispherical pit | radius = 10.0 | 0.8662 | 0.500 | ~19 ms |
| Sinusoidal waves | analytic height | 0.9094 | - | ~17 ms |
| Sinusoidal waves | from normals | 0.9087 | - | ~19 ms |
| Tilted plane | analytic height | 0.9897 | - | ~18 ms |
| Tilted plane | from normals | 0.9272 | - | ~21 ms |

## Verification and Limits

Accessibility stays strictly in [0.0, 1.0]. Flatter terrains remain close to 1.0, crevices and deep pits register progressive shadow obstruction. The metric is heightfield/horizon-based and does not compute full 3D mesh raytracing or volumetric ambient bounces.

