# Model card: bellium/knn/color-cutout:v0

- Task: RGB hard-edge foreground on a simple background; consultative only.
- Method: five nearest border samples, compacted by exact color multiplicity.
- Data: current image only; no trained weights or retained pixels.
- Limits: at most 512 distinct border colors; abstains beyond that budget or on
  implausible foreground area. Hair, glass and complex backgrounds are unsupported.
- Confidence: heuristic, not calibrated probability.
- Evidence: visual tests, exact-distance regression and `scripts/benchmark_baselines.py`.
- Independent silhouette pilot: exact masks on two controlled composites and
  abstention on low contrast. Photographic segmentation and RAM/VRAM remain unmeasured.
