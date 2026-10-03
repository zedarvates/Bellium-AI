# ApproxSurface + TopologyGrammar

Status: planned / experimental. This document defines a bounded deterministic-first contract for topology-aware 3D asset processing. It does not claim production-ready automatic retopology or UV unwrapping.

## Goal

Build a shared surface-analysis layer that can be reused by retopology, UV seam proposal, LOD, collision preparation and animation-oriented mesh validation.

Principle: do not ask a neural model to solve geometry that analytical or deterministic methods can establish sufficiently well and more cheaply.

## Pipeline

```
raw mesh
  -> surface measurements
  -> ApproxSurface descriptors
  -> semantic region hints
  -> TopologyGrammar pattern selection
  -> retopology / seam proposals
  -> deformation + UV validation
  -> optional escalation
```

The same measured surface graph should be reused across downstream stages instead of recomputing independent analyses.

## ApproxSurface v0 primitives

Each primitive returns:

- `value` or bounded estimate
- `interval` when meaningful
- `confidence`
- `evidence` / method
- `compute_cost`
- `latency_cost`
- `usable_for`
- `unsafe_for`
- `next_if_needed`

Initial primitives:

- `approx-curvature-v0`: local curvature / feature strength.
- `approx-surface-flow-v0`: dominant tangent directions and candidate polygon-flow directions.
- `approx-developability-v0`: local suitability for flattening with bounded distortion.
- `approx-seam-v0`: UV seam candidates with geometric and visibility evidence.
- `approx-density-v0`: suggested local polygon density from deformation, silhouette and detail needs.

Deterministic geometry remains the authority whenever confidence is sufficient.

## TopologyGrammar v0

TopologyGrammar is a library of reusable deformation patterns plus placement and validation rules.

Initial semantic patterns:

- `mouth-loop-v0`
- `eye-loop-v0`
- `nose-flow-v0`
- `jaw-neck-v0`
- `shoulder-v0`
- `elbow-v0`
- `knee-v0`
- `finger-joint-v0`
- `hip-groin-v0`

A pattern is not just a template mesh. It must declare:

- expected semantic anchors
- loop / edge-flow intent
- required degrees of freedom
- allowed poles / extraordinary vertices
- forbidden placements
- target density envelope
- expected deformation tests
- UV implications
- failure / abstention conditions

## Mouth pattern requirements

The first animation-critical pattern should be the mouth.

Required evidence before accepting a generated topology:

- concentric or functionally equivalent lip-support loops
- continuity toward cheeks, chin and nasolabial region
- sufficient vertices for lip closure, opening, smile, pucker and asymmetric motion
- no high-confidence self-intersection during test poses
- bounded stretch / compression around the oral commissures
- no seam placement through high-visibility lip surfaces unless explicitly requested

This is an animation constraint, not a cosmetic rule.

## UV objective

UV proposal should minimize a weighted cost rather than optimize only island count:

```
C = alpha * distortion
  + beta  * seam_visibility
  + gamma * semantic_importance
  + delta * fragmentation
```

Weights are profile-specific.

Suggested profiles:

- `game-general`: balance distortion, island count and packing.
- `vr-character`: prioritize close-view face/hands and memory cost.
- `story-closeup`: prioritize visible facial regions and shot-driven detail.
- `lod-background`: strongly reduce complexity and UV fragmentation.
- `print-no-uv`: skip UV work when texturing is not required.

## Escalation policy

Use a Parcimonia-style cascade:

1. analytical / deterministic geometry
2. heuristics + graph reasoning
3. k-NN / compact specialist
4. micro-NN only when measured benefit exists
5. larger vision / language model only for unresolved ambiguity

Every escalation must preserve the original measurements and explain why the cheaper stage abstained.

## Animation-aware density

Retopology should use intended animation as an input.

Examples:

- background NPC: minimal facial loops, limited expression set
- speaking NPC: stronger mouth / jaw / eyelid topology
- cinematic close-up: higher deformation fidelity around lips, cheeks, eyelids and brows
- rigid prop: no anatomical grammar; hard-surface rules instead

This lets StoryCore and game pipelines request topology according to actual use rather than using one mesh budget for every asset.

## First benchmark gate

Start with a small immutable corpus:

1. neutral humanoid head
2. stylized head
3. scan-like noisy head
4. shoulder/elbow limb fixture
5. rigid hard-surface control

For faces, include target poses:

- jaw open
- lip close
- smile
- pucker
- left/right asymmetric mouth motion
- blink

Measure:

- invalid topology count
- self-intersections
- vertex / triangle count
- deformation stretch and compression
- silhouette error
- UV distortion
- seam visibility proxy
- island count
- processing p50/p95 latency
- peak memory
- abstention rate
- expensive-model call rate

Do not claim superiority without comparing against deterministic baselines and at least one existing retopology / unwrap reference workflow.

## Integration targets

- Bellium AI: bounded specialists and benchmarks.
- AIMesher: consume surface descriptors and topology proposals through patchable geometry workflows.
- Ultimate Odycer Asset Factory: quality gates, LOD/UV/export orchestration.
- StoryCore Asset Creator: character-use profile and close-up / speaking requirements.
- Parcimonia: escalation and cost-aware routing.

## Non-goals for v0

- no autonomous rigging claim
- no full-body anatomical understanding claim
- no guarantee of film-quality facial topology
- no forced neural step
- no direct production activation before fixture evidence
