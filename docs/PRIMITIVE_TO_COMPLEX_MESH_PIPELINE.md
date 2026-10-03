# Primitive-to-Complex Mesh Pipeline

Status: planned / experimental. This document defines a deterministic-first construction pipeline for building editable 3D assets from simple primitives, concept guidance and surface patterns. It complements ApproxSurface + TopologyGrammar; it does not replace them.

## Pipeline

```
PrimitiveLibrary
  -> ShapeAssemblyGrammar
  -> ConceptGuide
  -> FormRefinement
  -> SurfacePatternSystem
  -> ApproxSurface
  -> TopologyGrammar
  -> UV / LOD / collision / export
```

The main design goal is to preserve editability, provenance and bounded cost. Complex geometry should emerge from inspectable operations rather than a single opaque mesh generation step whenever possible.

## 1. PrimitiveLibrary

Supported initial primitives:

- box
- cylinder
- sphere
- cone
- torus
- plane
- capsule
- polyline / spline
- extruded 2D profile
- lathe / revolution profile

Each primitive instance declares:

- transform
- dimensions
- semantic role
- symmetry
- local coordinate frame
- editable parameters
- provenance

## 2. ShapeAssemblyGrammar

ShapeAssemblyGrammar describes how semantic parts combine.

Initial object grammars:

- sword: blade + guard + grip + pommel
- chair: seat + back + supports / legs
- hammer: head + handle + optional reinforcement
- crate / chest: shell + lid + hinges + handle
- simple robot: torso + head + limb segments + joints
- stylized tree: trunk + branch hierarchy + foliage volumes
- modular wall: panel + frame + openings + repeated details

Grammar rules must remain declarative and inspectable. They may define required parts, optional parts, constraints, symmetry, attachment anchors and legal modifier families.

## 3. ConceptGuide

ConceptGuide treats images, sketches or concept art as constraints rather than unquestioned ground truth.

Extract or estimate:

- silhouette
- proportions
- dominant axes
- symmetry / asymmetry
- part placement
- characteristic negative spaces
- style descriptors
- high-importance detail regions

Each observation carries confidence and evidence. Ambiguous 2D-to-3D interpretation must remain explicit and may trigger abstention or multiple candidates.

Concept guidance should prefer low-cost geometric fitting before invoking a large model.

## 4. FormRefinement

Allowed deterministic operators initially:

- translate / rotate / scale
- extrude
- inset
- bevel
- bridge
- boolean union / difference / intersection
- bend
- twist
- taper
- smooth
- subdivision
- shrinkwrap / surface projection
- mirror / array
- sweep along curve

Every operation should be reversible or reconstructable from its recorded parameters where possible.

## 5. SurfacePatternSystem

Surface patterns are classified by representation level:

- texture-only: albedo / roughness / masks
- shading relief: normal / bump
- displacement: bounded geometric relief
- explicit geometry: large or silhouette-relevant details

Initial pattern families:

- wood grain
- woven cloth
- scales
- brushed metal
- panel lines
- rivets
- grooves
- stone veins
- seams / stitching
- organic ridges

The representation choice should depend on feature scale, silhouette importance, deformation needs, viewing distance and target platform.

## 6. New surface reasoning primitives

Add to the ApproxSurface family:

- `approx-feature-scale-v0`: classifies a detected feature as macro / meso / micro relative to asset size and target use.
- `approx-crease-v0`: detects folds, sharp transitions and persistent feature lines.
- `approx-silhouette-importance-v0`: estimates how strongly a region affects the viewed silhouette across representative viewpoints.

Suggested local importance objective:

```
importance =
    wc * curvature
  + ws * silhouette_importance
  + wd * deformation_importance
  + wa * semantic_importance
```

Weights are profile-specific and must be documented.

## 7. Anisotropic mesh density

Mesh density should not be a scalar only. Where supported, `approx-density-v0` should expose:

- density_u
- density_v
- principal_direction
- confidence
- reason

This allows elongated quads along low-change directions and denser edge flow across strong curvature, creases or deformation zones.

## 8. Construction profiles

Initial target profiles:

- `hard-surface-prop`
- `modular-environment`
- `stylized-organic`
- `humanoid-base-mesh`
- `creature-blockout`
- `printable-object`

Guidance:

- hard-surface and architecture should strongly favor primitives, profiles, booleans and bevels.
- humanoids should usually start from a semantic base mesh rather than raw primitives once deformation quality matters.
- creatures may use primitive volumetric scaffolds followed by semantic remeshing.
- printable objects may bypass UV/texturing entirely.

## 9. Reference-driven detail routing

Concept art and texture/pattern references can be used together.

Examples:

- silhouette reference drives primitive proportions.
- pattern reference drives material orientation and scale.
- large rivets become geometry.
- fine scratches remain texture / normal.
- cloth seams may become displacement or geometry depending on close-up needs.

The system must preserve the difference between inferred shape and directly measured geometry.

## 10. Provenance and patchability

Every construction step should produce a compact operation record:

- operation id
- input object ids
- operator
- parameters
- semantic intent
- source evidence
- confidence
- reversible flag
- resulting object ids

This lets AIMesher apply construction as patchable operations and supports STOP / inspect / correct / resume workflows.

## 11. First implementation gate

Start with three fixture families:

### Hard-surface fixture
Build a sci-fi hammer from primitives.
Measure:
- silhouette fit
- number of operations
- editable parameter count
- boolean / non-manifold failures
- final triangle count
- UV readiness

### Modular fixture
Build a wall panel with frame, opening, repeated grooves and rivets.
Measure:
- repeat consistency
- parameter reuse
- topology validity
- LOD simplification

### Stylized organic fixture
Build a simple trunk + branch structure from cylinders / curves.
Measure:
- branch continuity
- intersection quality
- silhouette error
- retopology burden

For every fixture compare:
1. deterministic primitive construction
2. primitive + ConceptGuide
3. primitive + ConceptGuide + SurfacePatternSystem
4. heavier generative fallback only when lower-cost stages abstain

Track p50/p95 latency, peak memory, operator count, manual corrections and expensive-model calls.

## 12. Integration targets

- Bellium AI: primitives, descriptors, pattern routing, bounded specialists and benchmarks.
- AIMesher: patchable construction graph and interactive corrections.
- Ultimate Odycer Asset Factory: orchestration, profiles, review gates, LOD/UV/export.
- StoryCore Asset Creator: concept-art-driven character/prop construction intents.
- Parcimonia: cost-aware routing and abstention/escalation.

## Non-goals for v0

- no claim of automatic production-quality character modeling
- no forced neural generation
- no hidden destructive modifier stack
- no conversion of every texture detail into geometry
- no automatic publication or merge into production assets
