Model card: bellium/knn/prosody-profile:v0

- Task: suggest whether a short syllable contour looks iambic, trochaic or even.
- Authority: consultative.
- Inputs: stress and duration per syllable, or named features. Raw audio is rejected.
- Certification: always false. A profile is not a recording and not a reconstructed melody.
- Limits: lab contours; unknown languages abstain; mixed neighbors abstain; missing durations fail closed.
- Evidence: tests/test_prosody.py.

