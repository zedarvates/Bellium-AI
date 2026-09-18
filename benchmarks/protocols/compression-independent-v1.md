# Independent compression gate v1 — fixed before measurement

Scope: one bounded experiment, no automatic model promotion or live integration.
Keep BLCP/1 and its pinned predictor vectors unchanged.

## Sources and split

- Development images: scikit-image v0.25.2 `camera.png` and `coffee.png`.
- Held-out images: the same release's `astronaut.png` and `chelsea.png`.
- Development stream: OurAirports `countries.csv`; held-out: `regions.csv`.
  Resolve a full repository commit before downloading either file.
- Held-out splats: the local Fovea Horse Statue CC0-derived trained asset with
  SHA256 `4392920074f41204ec900244da06505a8696312ea989340df88b1bd9c025806a`.
  Its source is a synthetic multiview video, not a physical camera capture.

Images: three 96x96 crops per source (top-left, center, bottom-right), no resize.
Streams: two 8192-byte development chunks at the beginning/end; three held-out
chunks at beginning/center/end. Splats: three 512-record subsets at those positions.
No source appears in both splits. All crops/subsets from a source remain grouped.
Small sample count and a single splat scene limit generalization.

The PLY adapter explicitly converts log scales, opacity logits and DC color,
normalizes and reorders rotations, then packs the supported 14 float32 fields.
It drops higher-order SH and extra fields. Exact BLCP reconstruction applies to
the canonical records only, **not** to the original complete PLY file. No natural
motion sequence is available for this gate.

## Measurements and one candidate

Run existing raw/zlib/delta/k-NN/micro-NN and automatic mode, with exact byte
reconstruction, complete packet size, and three timing repetitions. Include PNG
references for images. Do not tune codecs based on held-out results.

One hypothesis: the first micro-NN lacks vertical image context. Try one spatial
linear neuron with left/up/upper-left inputs and bias, Q12 weights, integer online
normalized updates, initial equal left/up weights, denominator factor 8. State
resets per color plane. Compare with fixed left/up/average/clipped-gradient
predictors using the same envelope and residual zlib. Preserve alpha/invisible RGB.

Development gate: exact reconstruction and aggregate packet bytes at least 1%
below the best fixed spatial method selected independently for each sample.
Only a candidate passing this gate gets a held-out evaluation. No parameter
sweep or second candidate in this run. Held-out promotion would additionally
require <=5% worst-case size regression, <=10x fixed-method median encode/decode
cost, and exact reconstruction everywhere. PNG comparisons remain explicit.

On rejection, retain the experimental result and reusable corpus tooling, leave
the production-facing codecs/catalog unchanged and state the negative result.
This gate does not establish .fovea superiority, VR performance or real-time use.
