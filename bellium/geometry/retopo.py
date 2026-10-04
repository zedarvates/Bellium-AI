"""Consultative retopology path proposals over an existing mesh.

This module does not rewrite topology. It proposes edge paths that can serve as
future loop/strip guides, preserving ambiguity and confidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from bellium.geometry.analyze import EdgeRecord, crease_records, edge_adjacency
from bellium.geometry.features import face_flow_hint
from bellium.geometry.mesh import Mesh, Vec3

Edge = tuple[int, int]


@dataclass(frozen=True)
class RetopoPath:
    edges: tuple[Edge, ...]
    score: float
    reason: str
    closed: bool
    ambiguous: bool


def _sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0]-b[0], a[1]-b[1], a[2]-b[2])


def _dot(a: Vec3, b: Vec3) -> float:
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]


def _norm(v: Vec3) -> float:
    return math.sqrt(_dot(v, v))


def _normalize(v: Vec3) -> Vec3:
    length = _norm(v)
    if length <= 1e-12:
        return (0.0, 0.0, 0.0)
    return (v[0]/length, v[1]/length, v[2]/length)


def edge_direction(mesh: Mesh, edge: Edge) -> Vec3:
    return _normalize(_sub(mesh.vertices[edge[1]], mesh.vertices[edge[0]]))


def _edge_faces(mesh: Mesh) -> dict[Edge, tuple[int, ...]]:
    return edge_adjacency(mesh)


def _candidate_score(mesh: Mesh, edge: Edge, faces: tuple[int, ...]) -> float:
    direction = edge_direction(mesh, edge)
    scores: list[float] = []
    for face in faces:
        flow = face_flow_hint(mesh, face)
        scores.append(abs(_dot(direction, flow.principal_direction)))
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def scored_edges(mesh: Mesh) -> tuple[tuple[Edge, float], ...]:
    mapping = _edge_faces(mesh)
    items = [
        (edge, _candidate_score(mesh, edge, faces))
        for edge, faces in mapping.items()
    ]
    return tuple(sorted(items, key=lambda item: (-item[1], item[0])))


def crease_paths(mesh: Mesh, threshold_degrees: float = 35.0) -> tuple[RetopoPath, ...]:
    records = crease_records(mesh, threshold_degrees)
    if not records:
        return ()
    adjacency_by_vertex: dict[int, list[EdgeRecord]] = {}
    for record in records:
        for vertex in record.edge:
            adjacency_by_vertex.setdefault(vertex, []).append(record)

    unused = {record.edge: record for record in records}
    paths: list[RetopoPath] = []

    while unused:
        edge, record = next(iter(unused.items()))
        del unused[edge]
        path = [edge]
        ambiguous = False

        for endpoint_index in (0, 1):
            current_vertex = path[0][0] if endpoint_index == 0 else path[-1][1]
            while True:
                candidates = [
                    item for item in adjacency_by_vertex.get(current_vertex, ())
                    if item.edge in unused
                ]
                if not candidates:
                    break
                if len(candidates) > 1:
                    ambiguous = True
                candidates.sort(
                    key=lambda item: (
                        -(item.dihedral_degrees or 0.0),
                        item.edge,
                    )
                )
                chosen = candidates[0]
                del unused[chosen.edge]
                a, b = chosen.edge
                oriented = (b, a) if b == current_vertex else (a, b)
                if endpoint_index == 0:
                    path.insert(0, oriented)
                    current_vertex = oriented[0]
                else:
                    path.append(oriented)
                    current_vertex = oriented[1]

        closed = len(path) > 2 and path[0][0] == path[-1][1]
        angle_strength = [
            item.dihedral_degrees / 180.0
            for item in records
            if item.edge in {tuple(sorted(e)) for e in path}
            and item.dihedral_degrees is not None
        ]
        score = sum(angle_strength) / len(angle_strength) if angle_strength else 1.0
        paths.append(
            RetopoPath(
                edges=tuple(path),
                score=max(0.0, min(1.0, score)),
                reason="crease-chain",
                closed=closed,
                ambiguous=ambiguous,
            )
        )
    return tuple(paths)


def flow_paths(mesh: Mesh, min_alignment: float = 0.85) -> tuple[RetopoPath, ...]:
    if not 0.0 <= min_alignment <= 1.0:
        raise ValueError("min_alignment must be between 0 and 1")
    accepted = {
        edge: score
        for edge, score in scored_edges(mesh)
        if score >= min_alignment
    }
    if not accepted:
        return ()

    vertex_edges: dict[int, list[Edge]] = {}
    for edge in accepted:
        vertex_edges.setdefault(edge[0], []).append(edge)
        vertex_edges.setdefault(edge[1], []).append(edge)

    unused = set(accepted)
    paths: list[RetopoPath] = []
    while unused:
        seed = min(unused)
        unused.remove(seed)
        path = [seed]
        ambiguous = False

        for endpoint_index in (0, 1):
            current = path[0][0] if endpoint_index == 0 else path[-1][1]
            while True:
                candidates = [edge for edge in vertex_edges.get(current, ()) if edge in unused]
                if not candidates:
                    break
                ranked = sorted(
                    candidates,
                    key=lambda edge: (-accepted[edge], edge),
                )
                if len(ranked) > 1 and abs(accepted[ranked[0]] - accepted[ranked[1]]) < 0.05:
                    ambiguous = True
                chosen = ranked[0]
                unused.remove(chosen)
                a, b = chosen
                oriented = (b, a) if b == current else (a, b)
                if endpoint_index == 0:
                    path.insert(0, oriented)
                    current = oriented[0]
                else:
                    path.append(oriented)
                    current = oriented[1]

        closed = len(path) > 2 and path[0][0] == path[-1][1]
        score = sum(accepted[tuple(sorted(edge))] for edge in path) / len(path)
        paths.append(
            RetopoPath(
                edges=tuple(path),
                score=score,
                reason="surface-flow",
                closed=closed,
                ambiguous=ambiguous,
            )
        )
    return tuple(paths)


def propose_retopo_paths(
    mesh: Mesh,
    *,
    crease_threshold_degrees: float = 35.0,
    min_alignment: float = 0.85,
) -> tuple[RetopoPath, ...]:
    proposals = list(crease_paths(mesh, crease_threshold_degrees))
    proposals.extend(flow_paths(mesh, min_alignment))
    return tuple(
        sorted(
            proposals,
            key=lambda path: (
                path.ambiguous,
                -path.score,
                path.reason,
                path.edges,
            ),
        )
    )
