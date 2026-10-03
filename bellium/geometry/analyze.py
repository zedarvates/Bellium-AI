"""Deterministic surface measurements for Bellium's minimal mesh kernel.

These are bounded geometric descriptors, not production retopology decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from bellium.geometry.mesh import Mesh, Vec3

Edge = tuple[int, int]


@dataclass(frozen=True)
class EdgeRecord:
    edge: Edge
    faces: tuple[int, ...]
    dihedral_degrees: float | None
    boundary: bool
    non_manifold: bool


@dataclass(frozen=True)
class SurfaceAnalysis:
    vertex_count: int
    face_count: int
    edge_count: int
    boundary_edge_count: int
    non_manifold_edge_count: int
    crease_edges: tuple[EdgeRecord, ...]
    face_normals: tuple[Vec3, ...]
    silhouette_edge_counts: tuple[tuple[str, int], ...]


def _sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0]-b[0], a[1]-b[1], a[2]-b[2])


def _cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1]*b[2]-a[2]*b[1],
        a[2]*b[0]-a[0]*b[2],
        a[0]*b[1]-a[1]*b[0],
    )


def _dot(a: Vec3, b: Vec3) -> float:
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]


def _norm(v: Vec3) -> float:
    return math.sqrt(_dot(v, v))


def _normalize(v: Vec3) -> Vec3:
    length = _norm(v)
    if length <= 1e-12:
        return (0.0, 0.0, 0.0)
    return (v[0]/length, v[1]/length, v[2]/length)


def face_normal(mesh: Mesh, face_index: int) -> Vec3:
    face = mesh.faces[face_index]
    if len(face) < 3:
        return (0.0, 0.0, 0.0)
    origin = mesh.vertices[face[0]]
    for i in range(1, len(face)-1):
        a = _sub(mesh.vertices[face[i]], origin)
        b = _sub(mesh.vertices[face[i+1]], origin)
        normal = _cross(a, b)
        if _norm(normal) > 1e-12:
            return _normalize(normal)
    return (0.0, 0.0, 0.0)


def edge_adjacency(mesh: Mesh) -> dict[Edge, tuple[int, ...]]:
    mapping: dict[Edge, list[int]] = {}
    for face_index, face in enumerate(mesh.faces):
        for i, start in enumerate(face):
            end = face[(i+1) % len(face)]
            edge = (start, end) if start < end else (end, start)
            mapping.setdefault(edge, []).append(face_index)
    return {edge: tuple(indices) for edge, indices in mapping.items()}


def dihedral_degrees(normal_a: Vec3, normal_b: Vec3) -> float:
    if _norm(normal_a) <= 1e-12 or _norm(normal_b) <= 1e-12:
        return 0.0
    value = max(-1.0, min(1.0, _dot(normal_a, normal_b)))
    return math.degrees(math.acos(value))


def edge_records(mesh: Mesh) -> tuple[EdgeRecord, ...]:
    normals = tuple(face_normal(mesh, i) for i in range(len(mesh.faces)))
    records: list[EdgeRecord] = []
    for edge, faces in sorted(edge_adjacency(mesh).items()):
        boundary = len(faces) == 1
        non_manifold = len(faces) > 2
        angle = None
        if len(faces) == 2:
            angle = dihedral_degrees(normals[faces[0]], normals[faces[1]])
        records.append(
            EdgeRecord(
                edge=edge,
                faces=faces,
                dihedral_degrees=angle,
                boundary=boundary,
                non_manifold=non_manifold,
            )
        )
    return tuple(records)


def crease_records(mesh: Mesh, threshold_degrees: float = 35.0) -> tuple[EdgeRecord, ...]:
    if threshold_degrees < 0.0 or threshold_degrees > 180.0:
        raise ValueError("crease threshold must be between 0 and 180 degrees")
    return tuple(
        record
        for record in edge_records(mesh)
        if record.non_manifold
        or record.boundary
        or (
            record.dihedral_degrees is not None
            and record.dihedral_degrees >= threshold_degrees
        )
    )


_VIEW_DIRECTIONS: tuple[tuple[str, Vec3], ...] = (
    ("+x", (1.0, 0.0, 0.0)),
    ("-x", (-1.0, 0.0, 0.0)),
    ("+y", (0.0, 1.0, 0.0)),
    ("-y", (0.0, -1.0, 0.0)),
    ("+z", (0.0, 0.0, 1.0)),
    ("-z", (0.0, 0.0, -1.0)),
)


def silhouette_edge_count(mesh: Mesh, view_direction: Vec3) -> int:
    direction = _normalize(view_direction)
    normals = tuple(face_normal(mesh, i) for i in range(len(mesh.faces)))
    count = 0
    for record in edge_records(mesh):
        if record.boundary or record.non_manifold:
            count += 1
            continue
        if len(record.faces) != 2:
            continue
        a = _dot(normals[record.faces[0]], direction)
        b = _dot(normals[record.faces[1]], direction)
        if (a < 0.0 <= b) or (b < 0.0 <= a):
            count += 1
    return count


def analyze_surface(mesh: Mesh, crease_threshold_degrees: float = 35.0) -> SurfaceAnalysis:
    records = edge_records(mesh)
    normals = tuple(face_normal(mesh, i) for i in range(len(mesh.faces)))
    crease = crease_records(mesh, crease_threshold_degrees)
    silhouettes = tuple(
        (name, silhouette_edge_count(mesh, direction))
        for name, direction in _VIEW_DIRECTIONS
    )
    return SurfaceAnalysis(
        vertex_count=len(mesh.vertices),
        face_count=len(mesh.faces),
        edge_count=len(records),
        boundary_edge_count=sum(record.boundary for record in records),
        non_manifold_edge_count=sum(record.non_manifold for record in records),
        crease_edges=crease,
        face_normals=normals,
        silhouette_edge_counts=silhouettes,
    )
