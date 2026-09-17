<p align="center">
  <img src="docs/assets/bellium-ai-banner.png" alt="Bellium AI banner" width="100%" />
</p>

# Bellium AI

[![Bellium AI CI](https://github.com/zedarvates/Bellium-AI/actions/workflows/ci.yml/badge.svg)](https://github.com/zedarvates/Bellium-AI/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-brightgreen.svg)](https://www.python.org/downloads/)

> **Small minds. Sharp purpose. Local intelligence.**

Bellium AI is an open-source laboratory for **micro-NN, k-NN, micro-LLM and hybrid specialist intelligence**: small, measurable models built to solve bounded recurring problems without calling a large general-purpose model for everything.

🌹 **Visual identity:** abyssal blue, crystalline intelligence and a blue rose at the core — a small touch of fantasy and mystery.

## Modules & Architecture

Bellium AI is structured around zero-heavy-dependency specialist primitives:

- [`bellium/cutout/`](bellium/cutout/) — Deterministic foreground segmentation, border sampling, alpha feathering and background normalization.
- [`bellium/inpaint/`](bellium/inpaint/) — Fast localized patch k-NN synthesis and mask-ratio escalation routing.
- [`bellium/language/`](bellium/language/) — Provenance-grounded phonemic vector space with epistemic evidence tracking (`attested`, `reconstructed`, `inferred`, `speculative`).
- [`bellium/routing/`](bellium/routing/) — Tiered specialist router (deterministic → k-NN → micro-NN → micro-LLM → large model) with hard constraint filtering and graceful abstention.
- [`bellium/pipeline/`](bellium/pipeline/) — End-to-end asset preparation chaining inpainting, cutout and canvas normalization for StoryCore / game engines.
- [`bellium/vision/`](bellium/vision/) — Deterministic visual anomaly, sensor state and emergency stop detection (robotics / aquaponics / CCTV).
- [`bellium/audio/`](bellium/audio/) — Zero-dependency voice activity detection (VAD), SNR estimation and clipping/quality gates.
- [`bellium/adapters/`](bellium/adapters/) — Deterministic JSON repair, schema enforcement, type coercion and micro-LLM output stabilization.
- [`legacy/`](legacy/) — Pinned weight-exact imports from `zedarvates/botte-secrete`: 11 micro-NN JSON models and family-isolated k-NN Asset Quality.

## Quick Start & Verification

```bash
# Clone repository
git clone https://github.com/zedarvates/Bellium-AI.git
cd Bellium-AI

# Run the 10 validation suites (cross-platform, pure Python / Pillow)
python run_all_tests.py
```

## Usage examples: tools, scripts and an agent skill

See [the runnable examples](examples/README.md) for a local image CLI, a callable
tool adapter, tool routing, imported micro-NN error triage and a reusable
[Bellium skill](examples/skills/bellium-local-tools/SKILL.md).

```sh
python -m pip install -r examples/requirements.txt
python -m examples.tools.image_tool demo --output-dir output/bellium-demo
python -m examples.tool_routing
python -m examples.micro_nn_triage
```

The image demo creates synthetic inputs and separate grayscale, sepia, cutout
and small-fill previews. Filters use Pillow; cutout/fill use the published
Bellium primitives. Original files are preserved and uncertain requests can
abstain. Physics, PBR extraction, nano-NNs and a full image editor remain
[roadmap items](docs/ROADMAP.md).

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

## Existing specialists to migrate

The first migration preserves provenance from `zedarvates/botte-secrete` rather than replacing its live integrations. It includes the existing micro-NN family and the shadow-only Asset Quality k-NN algorithm. Private/local neighbor memories are not published.

See [`docs/MIGRATION_INVENTORY.md`](docs/MIGRATION_INVENTORY.md) and [`docs/ROADMAP.md`](docs/ROADMAP.md).

## Evidence and authority

Every specialist records its task, data provenance, baseline, hardware, latency, RAM/VRAM, quality metrics, dangerous false positives, abstention, escalation and known limits.

Authority modes are explicit:

`observe → consultative → shadow → active`

Importing or benchmarking a model never silently promotes its authority.

---

> **Bellium AI — intelligence distilled until only the useful signal remains.**
