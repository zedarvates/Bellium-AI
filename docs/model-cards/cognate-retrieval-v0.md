Model card: bellium/knn/cognate-retrieval:v0

- Task: retrieve related word forms in other lects of the same family.
- Authority: consultative.
- Method: labelled cognate sets first, then form/phone k-NN. Same-language items are excluded.
- Certification: always false. Relatedness is not a reconstructed etymon.
- link_attested: true only when every used member of a labelled set is attested.
- Limits: lab family only; mixed glosses abstain; no audio; no proto-form.
- Evidence: tests/test_cognates.py.

