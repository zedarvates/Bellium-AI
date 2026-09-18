Model card: bellium/hybrid/panel-safe-crop:v0

- Task: propose a crop that keeps required panel content.
- Authority: consultative.
- Order: required boxes must stay inside the crop; k-NN only proposes a family margin.
- Inputs: image, layout family, optional integer content boxes. Missing box fields fail closed.
- Limits: not a comic parser; busy backgrounds without boxes abstain; already-clipped boxes are unsafe.
- Evidence: tests/test_panel_crop.py.

