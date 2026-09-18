Model card: bellium/hybrid/voice-activity:v0

- Task: label a short clip as speech or silence.
- Authority: consultative.
- Inputs: numeric samples or named frame features. Paths and stored waveforms are rejected.
- Method: energy, zero-crossings, periodicity; k-NN; optional window vote.
- Limits: not ASR, not diarization; audio is discarded after features.
- Evidence: tests/test_voice_activity.py.

