# Contributing

Keep each specialist bounded and explicit about evidence, abstention and authority.
Use `bellium.resources.model_path` for packaged model data. Preserve pinned weight
bytes and imported notices. Do not import private examples or operational ledgers.

`[tool.ruff.lint]` pins the lint rule set, because ruff's defaults changed across
releases while CI installs an unpinned `ruff>=0.6`. Widen the set deliberately and
clean the whole tree in the same change.

Before submitting a change, run `python -m ruff check .`, `python -m pytest -q` and
`python -m build`. Install the resulting wheel and run the installed-package smoke
check with `python -I scripts/smoke_installed.py`; add `--image` when Pillow is available.
Document changes to public behavior and include meaningful regression cases.

Synthetic rule imitation is not evidence that a neural network outperforms the
rule. Compare quality, abstention, false agreements and cost on independent data
before promoting a specialist beyond consultative or shadow mode.
