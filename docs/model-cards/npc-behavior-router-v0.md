Model card: bellium/micro-nn/npc-behavior-router:v0 and bellium/hybrid/npc-behavior-router:v0

- Task: propose a bounded behaviour label (idle, patrol, investigate, engage, retreat) from
  caller-declared compact state.
- Authority: consultative. executes stays false. The hybrid never launches movement, combat,
  networking or animation.
- Order: deterministic vetoes first (incapacitated or zero health returns down; noncombatant or
  surrendered returns hold_fire before any model is loaded). Declared constraints then block
  engage, and a proposal that contradicts a constraint abstains instead of being remapped.
- Micro-NN: layers [8, 12, 5], labels as above, trained by
  scripts/train_npc_behavior_router.py from an authored weighted danger/safety rule with a
  declared 5% label-noise rate, seed 17. The network imitates that rule; it is not a policy.
- Independent evaluation data: 300 clean-rule samples from a separate seed; held-out accuracy
  0.907. No gameplay session, player trace or engine replay was evaluated.
- Invalid input: missing or extra state features and unknown constraint keys raise. Unknown
  constraints are never ignored.
- Limits: no pathfinding, no learning online, no memory of episodes, no squad logic, no
  animation, no engine integration and no security or combat authority. Deterministic gameplay
  code still owns every action.
- Evidence: tests/test_npc_behavior.py.
