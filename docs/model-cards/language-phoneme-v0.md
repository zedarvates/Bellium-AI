Model card: bellium/knn/language-phoneme:v0

- Task: retrieve the nearest labelled phones in one language.
- Authority: consultative.
- Inputs: language id plus distinctive features. Missing formants stay missing.
- Outputs: proposed IPA, evidence class, neighbors, certification flag.
- Evidence classes: attested, reconstructed, inferred, speculative.
- Certification requires a unique exact feature match with attested support. Tied
  phones abstain; a non-exact neighbor is an inference, not an attested prediction.
- Limits: fixture inventory only; not a reconstructed language; not audio ASR.
- Evidence: tests/test_language_knn.py.
