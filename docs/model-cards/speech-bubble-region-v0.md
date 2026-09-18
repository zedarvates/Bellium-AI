Model card: bellium/hybrid/speech-bubble-region:v0

- Task: propose speech-bubble regions on a simple panel.
- Authority: consultative.
- Method: bright compact connected components, then k-NN bubble vs not_bubble.
- Outputs: boxes only. text is always null. Dialogue is never read or stored.
- Limits: not OCR, not a comic parser; dark or busy panels abstain.
- Evidence: tests/test_speech_bubble.py.

