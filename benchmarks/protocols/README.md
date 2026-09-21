# Evaluation protocol

Freeze the task, dataset revision/hash, source commit, seeds, software versions,
hardware, warm/cold policy, quality threshold, false-agreement definition and
abstention policy before evaluating a specialist. Include a deterministic baseline.

`python scripts/benchmark_baselines.py --output <report.json>` measures bounded
synthetic cutout latency and imitation of the routers' labeling rules. It does
not measure quality on real assets, end-to-end tool selection, RAM/VRAM, audio or
real aquaponics data. Record these missing measurements explicitly.

For useful release evidence, add independent licensed examples, hold them out
from training, and report quality/latency alongside the cost of the simpler rule.
Never replace an accepted model before the candidate has passed its gate.
