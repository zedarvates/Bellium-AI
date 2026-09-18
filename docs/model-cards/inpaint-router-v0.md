# Model card: bellium/micro-nn/inpaint-router:v0

- Task: decide whether a masked region is simple enough for patch k-NN inpainting.
- Authority: consultative.
- Inputs: six bounded mask/image statistics, no raw pixels.
- Outputs: patch_knn or escalate, with probabilities and abstention.
- Training: synthetic hole statistics, stdlib SGD, seed 11.
- Limits: not a painter; not valid on hair, glass, text, or large missing regions.
- Evidence: tests/test_visual_specialists.py.
- Its probability is a synthetic-rule label score, not reconstruction quality.
  Downstream patch filling now runs independent local-context probes and may abstain
  even when this router proposes patch_knn. Weights are unchanged.
