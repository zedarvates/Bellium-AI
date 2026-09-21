Model card: bellium/micro-nn/consequence-predictor:v0 and bellium/hybrid/consequence-predictor:v0

- Task: estimate what a proposed action would do: safe, review or dangerous.
- Authority: consultative. executes stays false. The specialist predicts, it never acts.
- Order: deterministic veto first. An action declared irreversible without a declared dry run
  or backup returns veto / dangerous before any model is loaded. Then verified precedents and
  the micro-NN are consulted.
- Reference: bellium/deterministic/consequence-risk:v0, a published weighted risk score with
  two thresholds. Every result reports baseline and agrees_with_baseline.
- Agreement: the micro-NN and the precedents must both reproduce the published rule, otherwise
  the result abstains with learned_tier_differs_from_rule. A learned tier that contradicts the
  rule produces an abstention, not a promotion.
- Micro-NN: layers [8, 16, 3], trained by scripts/train_consequence_predictor.py from the
  published rule with a declared 5% label-noise rate, seed 23, 2000 samples.
- Independent evaluation data: 300 held-out vectors from a separate seed. The rule scores 1.000
  by construction because it generates the labels; the measured quantity is the network
  agreement, 0.900. The gap is recorded, not hidden, and the rule stays the reference.
- Invalid input: missing or extra state features and unknown constraint keys raise. Unknown
  constraints are never ignored.
- Safety contract: all three constraints must be explicit booleans before learned
  tiers run; absent or contradictory declarations abstain unless a conservative veto
  already applies. A safe estimate still requires caller confirmation. See
  docs/CONSEQUENCE_CONTRACT.md for the explicit compatibility migration.
- Limits: no execution, no sandboxing, no rollback, no ledger of real incidents and no
  multi-user coordination. Gameplay and infrastructure policy still own every action.
- Evidence: tests/test_consequence.py.
