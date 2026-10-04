# Historical geometry kernel reused by ApproxSurface / TopologyGrammar

This track reconnects earlier AIMesher / NBF geometry research with the current
ShapeGrammar → ApproxSurface → retopology pipeline.

## What the older work was actually doing

The earlier research was broader than "average nearby points":

1. local point / surface projection and smoothing;
2. Surface Nets / Dual Contouring style placement;
3. QEF (Quadric Error Function) for feature-preserving vertex placement;
4. QEM (Quadric Error Metrics) for simplification under geometric constraints;
5. protection of creases, silhouette, openings, materials and topology;
6. low-poly / LOD generation after a denser reconstruction stage.

The current retopology work can reuse those ideas without pretending that one
algorithm solves the entire problem.

## Where each primitive belongs now

### Nearest-neighbour projection

Use after a new strip/loop vertex is proposed.

Purpose:

- pull a candidate vertex back toward the dense reference surface;
- provide a cheap deterministic baseline before exact triangle projection;
- expose confidence / radius limits and abstain when reference points are too far.

The first implementation is an inverse-distance weighted centroid of nearby
reference vertices. This is deliberately simple and transparent.

### QEF placement

Use when several local plane/normal constraints describe a crease, corner or
surface intersection.

Purpose:

- place a representative vertex where local planes agree;
- preserve corners/creases better than a plain point average;
- abstain on singular/underdetermined constraints;
- optionally clamp the result to a bounded cell/region.

This reconnects directly to the earlier Dual Contouring / NBF experiments.

### QEM simplification

Use after a valid topology exists, especially for LOD or controlled reduction.

Purpose:

- evaluate edge-collapse candidates;
- minimize geometric error;
- refuse protected feature edges;
- later incorporate silhouette, semantic, material and topology penalties.

QEM belongs after topology creation/validation, not as a replacement for
TopologyGrammar around animation-critical regions.

## Updated retopology pipeline

```
dense / source mesh
  -> ApproxSurface measurements
  -> flow / crease / feature-scale hints
  -> guide paths
  -> quad-strip proposal
  -> new candidate vertices
  -> nearest-neighbour projection baseline
  -> QEF refinement where local planes/normals justify it
  -> topology validation
  -> TopologyGrammar animation rules
  -> UV / seams
  -> QEM-constrained simplification for LOD
```

## Important separation

- nearest-neighbour centroid: cheap projection baseline;
- QEF: feature-preserving vertex placement;
- QEM: simplification / collapse scoring;
- TopologyGrammar: semantic/deformation rules;
- ApproxSurface: measurements and confidence.

These are complementary, not interchangeable.

## Next gate

1. create a strip with inserted midpoint vertices;
2. project those vertices to a dense reference mesh using neighbour centroid;
3. compare against exact nearest-triangle projection;
4. add QEF only in crease/corner fixtures;
5. benchmark point error, normal error, silhouette error and abstention;
6. only then enable QEM collapse experiments on non-protected regions.
