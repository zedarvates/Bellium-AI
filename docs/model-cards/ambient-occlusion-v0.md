Model card: bellium/hybrid/ambient-occlusion:v0

- Task: calculate horizon-based ambient occlusion accessibility factors in [0.0, 1.0] from a relative height field or normal map.
- Authority: consultative and inert. written_files, certified and pixels_changed stay false, false, and 0: accessibility values are returned as in-memory data.
- Inputs: height map OR normal field, optional mask, sampling radius (default 6), directions count (default 8), pixel_scale (default 1.0), elevation_scale (default 1.0).
- Output: 2D accessibility factor grid (ao), mean_ao across valid pixels, and diagnostic metadata.
- Method: ray-marching along discrete radial horizon angles (horizon elevation angle psi = atan(dh/dist)), projecting solid angle occlusion = sin(psi).
- Measured: benchmarks/manifests/ambient-occlusion-v1.json. V-grooves deepen from mean 0.835 (slope 0.5) to 0.661 (slope 2.0). Hemispherical pit centers reach strong occlusion (~0.50), while planar surfaces stay near 1.0.
- Limits: pure screen-space/heightfield horizon occlusion without global interreflections. Multi-bounce diffuse ambient bounces are not modeled.
- Evidence: tests/test_ambient_occlusion.py, scripts/benchmark_ambient_occlusion.py, docs/AMBIENT_OCCLUSION_GATE.md.

