Model card: bellium/hybrid/tool-router:v0

- Task: propose none, inspect_files, run_tests, git_status or escalate.
- Authority: consultative. No tool is executed.
- Order: deterministic veto, closed-catalog k-NN, micro-NN none/use_tool/escalate.
- Inputs: six named features and boolean risk signals. Raw text is rejected.
- Limits: mutation, secrets, deploy and unknown tools always escalate.
- Evidence: tests/test_tool_and_memory.py. Micro-NN trained on synthetic features.

