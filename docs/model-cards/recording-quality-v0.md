Model card: bellium/hybrid/recording-quality:v0

- Task: label a short clip as clean, noisy or clipped.
- Authority: consultative.
- Inputs: numeric samples or named frame features. Paths and stored waveforms are rejected.
- Method: energy, clipping, noise floor, periodicity, zero-crossings; k-NN.
- Limits: not denoising, not repair, not loudness certification; audio is discarded after features.
- Evidence: tests/test_recording_quality.py.

