Model card: bellium/hybrid/tool-router:v0

- Task: propose none, inspect_files, run_tests, git_status or escalate.
- Authority: consultative. No tool is executed.
- Order: deterministic veto, closed-catalog k-NN, micro-NN none/use_tool/escalate.
- Inputs: six named features and boolean risk signals. Raw text is rejected.
- Neural training domain: has_code, has_files, has_error, wants_mutation and
  wants_external_state are binary (0/1 or booleans); criticality is continuous
  in [0,1]. Fractional flags return a consultative abstention with
  `reason: outside_training_domain` before model loading. The numeric input
  schema still accepts those values for deterministic veto and k-NN processing.
  The hybrid preserves the deterministic veto before any neural domain check.
- Limits: mutation, secrets, deploy and unknown tools always escalate.
- Evidence: tests/test_tool_and_memory.py. Micro-NN trained on synthetic features.
- Domain evidence: tests/test_tool_router_domain.py and
  [the native training gate](../TOOL_ROUTER_TRAINING_GATE.md). Softmax confidence
  is uncalibrated. Published weights are unchanged by this gate.
- Explicit score calibration: `classify_tool_need(query, calibration=artifact)`
  and `route_tool(query, calibration=artifact)` accept a model-bound temperature
  artifact. No artifact is loaded automatically. Deterministic veto and domain
  abstention still precede it. The [native calibration gate](../NATIVE_CALIBRATION_GATE.md)
  rejected both synthetic candidates; default scores and threshold remain unchanged.
