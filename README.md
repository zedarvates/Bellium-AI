<p align="center">
  <img src="docs/assets/bellium-ai-banner.png" alt="Bellium AI banner" width="100%" />
</p>

# Bellium AI

> **Small minds. Sharp purpose. Local intelligence.**

Bellium AI is an open-source laboratory for **micro-NN, k-NN, micro-LLM and hybrid specialist intelligence**: small, measurable models built to solve bounded recurring problems without calling a large general-purpose model for everything.

🌹 **Visual identity:** abyssal blue, crystalline intelligence and a blue rose at the core — a small touch of fantasy and mystery.

## Mission

Use the **smallest competent mechanism** for each task and escalate only when needed.

Priority order when reasonable:

1. deterministic algorithm / classical CV or DSP
2. rules
3. k-NN / nearest exemplars
4. embeddings + k-NN
5. shallow classifier
6. micro-NN
7. specialist micro-LLM
8. compact local general model
9. large local or remote model

Abstention is a feature. Critical safety constraints stay deterministic.

## First project-driven tracks

- selected-zone image filling and compact inpainting
- image cutout / foreground extraction
- white or transparent background normalization
- asset-quality and near-duplicate detection
- language preservation: phonemes, pronunciation and grapheme↔phoneme reconstruction
- tool / skill / workflow routing
- memory reranking and anomaly detection
- robotics and aquaponics visual-state specialists
- StoryCore / game-asset preparation helpers

## Imported specialists

The first migration preserves provenance from `zedarvates/botte-secrete` rather than replacing its live integrations. It includes the existing micro-NN family and the shadow-only Asset Quality k-NN algorithm. Private/local neighbor memories are not published.

See [`docs/MIGRATION_INVENTORY.md`](docs/MIGRATION_INVENTORY.md) and [`docs/ROADMAP.md`](docs/ROADMAP.md).

## Evidence and authority

Every specialist must document its task, provenance, baseline, quality, dangerous false agreements,
abstention and limits before deployment. Current v0 evidence is primarily synthetic; real-data
quality and RAM/VRAM are not established. See [the evaluation protocol](benchmarks/protocols/README.md).

Authority modes are explicit:

`observe → consultative → shadow → active`

Importing or benchmarking a model never silently promotes its authority.

---

> **Bellium AI — intelligence distilled until only the useful signal remains.**

## Install and verify

```sh
python -m pip install .
# Optional Pillow image adapters:
python -m pip install ".[image]"
# Development and unified tests, including the retained published scenarios:
python -m pip install ".[dev]"
python -m ruff check .
python -m pytest -q
python -m build
```

Models are included in the wheel. Test the installed package outside the source
import path with `python -I scripts/smoke_installed.py --image` after installing it.

```python
from bellium.specialists.tool_router import route_tool

result = route_tool({"signals": {"mentions_secret": True}})
assert result.output["tool"] == "escalate"
# A proposal is returned; no tool is executed.
```

See [architecture and compatibility](docs/ARCHITECTURE.md) and
[the audit correction record](docs/STABILIZATION.md) for the merged APIs and behavior changes.

The [photographic evaluation](docs/VISUAL_QUALITY_V2.md) now tests an uncertainty
check on known neighboring regions. It reduced severe accepted errors from 7/24
to 0/11 on the same new test cases, but coverage remains below the 80% gate.
Photographic inpainting remains experimental; rejected candidates must not be applied.

## Local specialist snapshot

The first local implementations now live in this working tree:

- 11 imported Botte Secrete micro-NNs, observe-only, with SHA-256 provenance
- native patch k-NN inpainting
- native color k-NN cutout
- native asset-quality k-NN (shadow, no private memory)
- native inpaint-router micro-NN
- white-background hybrid built on the cutout

- native phoneme k-NN and pronunciation similarity

- native tool router (veto + k-NN + micro-NN)
- native memory reranker

- native visual anomaly k-NN (not the imported log detector)

- native grapheme-phoneme k-NN (second wave)

- native cognate retrieval k-NN (second wave)

- native prosody profile k-NN (second wave)

- native panel-safe crop (second wave)

- native speech-bubble region finder (second wave)

- native consistency retrieval k-NN (third wave)

- native flat-color helper (third wave)

- native voice-activity k-NN (third wave)

- native recording-quality k-NN (third wave)

See [docs/SPECIALISTS.md](docs/SPECIALISTS.md). No specialist is active by default.

Fourth wave, game-oriented:

- native texture-repeat and texture-tileability k-NN
- native nano-NN tile-seam classifier with a hard 64-parameter budget
- native sprite-anchor k-NN and bounded sprite frame preparation
- native NPC behaviour micro-NN with deterministic vetoes

See [the model tiers](docs/MODEL_TIERS.md) for the deterministic, k-NN, nano-NN
and micro-NN boundaries, and the nano size contract.

Fifth wave, sprite sheets and action consequences:

- native clip-loop k-NN and a sprite-sheet preparation report
- frame-phase tier measured and deliberately not shipped (the rule wins)
- native consequence precedents, micro-NN and hybrid with a deterministic veto

Sixth wave, physical estimates (P7 first gate):

- published references for gravity, atmosphere, pressure and ballistic previews
- k-NN case interpolation, a four-parameter nano gravity model and a 21-parameter
  micro atmosphere model, measured against the references in
  [the gate record](docs/PHYSICAL_ESTIMATES.md)

Seventh wave, texture grain (P8 first gate):

- native texture-orientation k-NN: grain direction, angle and period, with a perspective
  refusal instead of one global period
- shift-difference period measurement shared with the repeat and axis tests
- held-out distorted-texture measurement in
  [the gate record](docs/TEXTURE_REPEAT_GATE.md)

Eighth wave, animation timing:

- native easing-profile k-NN over the published curves, and a hybrid timing report with
  holds, peak, duplicates and warnings
- no neural tier retained: the measured candidate lost to the published rule
  ([gate record](docs/ANIMATION_TIMING_GATE.md))

Ninth wave, albedo and illumination separation (P8 second gate):

- controlled renders with known maps, a log-domain separation, and a specialist that
  returns both candidates and requires a declared shading prior
- measured finding: an image-only choice is right in 76 % of cases
  ([gate record](docs/MATERIAL_SEPARATION_GATE.md))

Tenth wave, normals from multi-light captures (P8 third gate):

- photometric stereo on controlled captures with declared light directions, per-patch
  trust from the fit residual, and abstention when the Lambertian model fails
- trusted patches at 0.0094 degrees against 7.86 degrees for the rejected ones
  ([gate record](docs/PHOTOMETRIC_NORMALS_GATE.md))

Eleventh wave, direct image editing (P9 first gate):

- seven deterministic filters with adjustable strength, feathered selection masks and a
  replayable edit stack with undo and redo
- magic eraser wiring that reuses the inpaint router and escalates when unsure
- measured 4 to 18 ms per 64 x 64 preview: not interactive yet
  ([gate record](docs/EDITING_GATE.md))

Twelfth wave, interactive editing path:

- vectorised, lookup-table and plain filter backends kept pixel-identical, plus
  area-averaged previews that are never exports
- a preview session at 14.7 ms p95 on a 512 x 512 source, against a cold one-shot
  preview of about 0.5 s ([interactive record](docs/EDITING_INTERACTIVE_GATE.md))

P4 gate, atlas packing:

- deterministic atlas geometry: bottom-left skyline placement with a padding gutter, a bounded
  page count and a plan that reports unplaced frames instead of shrinking them
- 10 authored cases x 3 methods with 0 invariant violations, measured against the two published
  shelf baselines ([gate record](docs/ATLAS_PACKING_GATE.md))

Local [compression research baselines](docs/model-cards/compression-v0.md) now
cover images, byte streams and static/temporal Gaussian splat records. They
compare causal k-NN and an adaptive linear micro-NN against classical codecs,
preserving exact bytes. Real-asset superiority and runtime integration are unproven.
