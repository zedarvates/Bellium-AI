# Model tiers

Bellium tiers are project size labels, not standardized architectures and never
language models. The smallest competent mechanism is tried first, and a larger
tier only earns its place with a measured benefit over the simpler one.

| Tier | Definition | Published example |
| --- | --- | --- |
| deterministic | Formula, threshold, classical CV/DSP or a 2D drafting document | `bellium/deterministic/drafting-document:v0` |
| k-NN | Family-local exemplar memory plus explicit features, no gradient training | `bellium/knn/texture-tileability:v0` |
| nano-NN | Hard parameter and byte budget, declared precision, bounded per-axis decisions | `bellium/nano-nn/tile-seam:v0` (38 parameters, 1816 bytes) |
| micro-NN | Small feed-forward classifier for a bounded recurring decision | `bellium/micro-nn/npc-behavior-router:v0` (173 parameters) |
| hybrid | Deterministic gates, then k-NN, then a small network; contradictions abstain | `bellium/hybrid/texture-tile-fixer:v0` |
| micro-LLM | Reserved for real small language models; not implemented | none |

## Nano tier contract

`bellium.nano_nn.contract` measures a model before it is used: parameter count
from the declared layer shapes, serialized bytes as indented JSON with LF
newlines, and the declared precision. A model over its declared budget raises
instead of loading, and `NanoBudget.describe()` reports `device_verified`,
`measured_latency_ms` and `measured_peak_ram_bytes` so a missing measurement
cannot be mistaken for a passing one.

Declared budgets today:

| Specialist | Parameters | Budget | Bytes | Budget |
| --- | --- | --- | --- | --- |
| `nano-tile-seam-v0` | 38 | 64 | 1816 | 8192 |

Measured on this workstation (Windows, CPython 3.14): warm prediction p50
0.013 ms, p95 0.014 ms over 2000 calls; full `classify_seam` including model
load and budget check p50 0.177 ms, p95 0.259 ms over 50 calls.

Not established: target-device latency, peak RAM, VRAM, quantized deployment,
and any advantage on real textures. An abstaining model is a valid outcome.

## When a tier is measured and not shipped

`bellium/nano-nn/frame-phase:v0` is the counter-example that keeps the tiers
honest. The task (animation phase from compact deltas) is already decided by the
published threshold rule `bellium/deterministic/phase-threshold:v0`. A
[7, 4, 4] candidate was trained and measured at 0.970 agreement against 1.000 for
the rule on 200 held-out transitions, so no weights were written and the
specialist answers `baseline_only` with `model_shipped: false`. The budget
(64 parameters, 8192 bytes) stays ready for a future task where a network
actually beats its baseline.
