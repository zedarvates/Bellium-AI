"""Deterministic feature-scale and anisotropic-density heuristics.

These functions provide bounded descriptors for later retopology stages. They do
not place quads, create UV seams, or infer anatomy.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from bellium.geometry.analyze import EdgeRecord, edge_records, face_normal
from bellium.geometry.export import extent
from bellium.geometry.mesh import Mesh, Vec3


@dataclass(frozen=True)
class FeatureScale:
    edge: tuple[int, int]
    length: float
    relative_length: float
    category: str


@dataclass(frozen=True)
class FlowHint:
    face_index: int
    principal_direction: Vec3
    secondary_direction: Vec3
    anisotropy: float


@dataclass(frozen=True)
class DensityHint:
    face_index: int
    density_u: float
    density_v: float
    principal_direction: Vec3
    reason: str


def _sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0]-b[0], a[1]-b[1], a[2]-b[2])


def _dot(a: Vec3, b: Vec3) -> float:
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]


def _norm(v: Vec3) -> float:
    return math.sqrt(_dot(v, v))


def _normalize(v: Vec3) -> Vec3:
    n = _norm(v)
    if n <= 1e-12:
        return (0.0, 0.0, 0.0)
    return (v[0]/n, v[1]/n, v[2]/n)


def _cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1]*b[2]-a[2]*b[1],
        a[2]*b[0]-a[0]*b[2],
        a[0]*b[1]-a[1]*b[0],
    )


def edge_length(mesh: Mesh, edge: tuple[int, int]) -> float:
    return _norm(_sub(mesh.vertices[edge[1]], mesh.vertices[edge[0]]))


def asset_scale(mesh: Mesh) -> float:
    dims = extent(mesh)
    diagonal = math.sqrt(dims[0]*dims[0] + dims[1]*dims[1] + dims[2]*dims[2])
    if diagonal <= 1e-12:
        raise ValueError("asset scale is zero")
    return diagonal


def classify_feature_scale(mesh: Mesh, record: EdgeRecord) -> FeatureScale:
    length = edge_length(mesh, record.edge)
    relative = length / asset_scale(mesh)
    if relative < 0.02:
        category = "micro"
    elif relative < 0.10:
        category = "meso"
    else:
        category = "macro"
    return FeatureScale(
        edge=record.edge,
        length=length,
        relative_length=relative,
        category=category,
    )


def feature_scales(mesh: Mesh) -> tuple[FeatureScale, ...]:
    return tuple(classify_feature_scale(mesh, record) for record in edge_records(mesh))


def _project_to_tangent(vector: Vec3, normal: Vec3) -> Vec3:
    amount = _dot(vector, normal)
    return (
        vector[0] - amount * normal[0],
        vector[1] - amount * normal[1],
        vector[2] - amount * normal[2],
    )


def face_flow_hint(mesh: Mesh, face_index: int) -> FlowHint:
    face = mesh.faces[face_index]
    if len(face) < 3:
        raise ValueError("face must have at least three vertices")
    normal = face_normal(mesh, face_index)
    candidates: list[tuple[float, Vec3]] = []
    for i, start in enumerate(face):
        end = face[(i + 1) % len(face)]
        vector = _sub(mesh.vertices[end], mesh.vertices[start])
        tangent = _project_to_tangent(vector, normal)
        length = _norm(tangent)
        if length > 1e-12:
            candidates.append((length, _normalize(tangent)))
    if not candidates:
        raise ValueError("face has no measurable tangent direction")
    candidates.sort(key=lambda item: item[0], reverse=True)
    primary_length, primary = candidates[0]

    secondary = (0.0, 0.0, 0.0)
    secondary_length = 0.0
    for length, direction in candidates[1:]:
        if abs(_dot(primary, direction)) < 0.95:
            secondary = direction
            secondary_length = length
            break
    if secondary_length <= 1e-12:
        secondary = _normalize(_cross(normal, primary))
        secondary_length = primary_length

    ratio = primary_length / secondary_length if secondary_length > 1e-12 else 1.0
    return FlowHint(
        face_index=face_index,
        principal_direction=primary,
        secondary_direction=secondary,
        anisotropy=max(1.0, ratio),
    )


def face_curvature_proxy(mesh: Mesh, face_index: int) -> float:
    """Average adjacent-face normal change in [0, 1]."""
    records = edge_records(mesh)
    own = face_normal(mesh, face_index)
    angles: list[float] = []
    for record in records:
        if face_index not in record.faces or len(record.faces) != 2:
            continue
        other_index = record.faces[0] if record.faces[1] == face_index else record.faces[1]
        other = face_normal(mesh, other_index)
        dot = max(-1.0, min(1.0, _dot(own, other)))
        angles.append(math.acos(dot) / math.pi)
    if not angles:
        return 0.0
    return sum(angles) / len(angles)


def density_hint(
    mesh: Mesh,
    face_index: int,
    *,
    base_density: float = 1.0,
    deformation_importance: float = 0.0,
    silhouette_importance: float = 0.0,
) -> DensityHint:
    if base_density <= 0.0:
        raise ValueError("base_density must be > 0")
    for value, name in (
        (deformation_importance, "deformation_importance"),
        (silhouette_importance, "silhouette_importance"),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")

    flow = face_flow_hint(mesh, face_index)
    curvature = face_curvature_proxy(mesh, face_index)
    importance = 1.0 + 2.0 * curvature + deformation_importance + silhouette_importance

    # Long direction needs fewer cuts than the cross direction. Clamp the ratio
    # so this remains a conservative hint rather than an aggressive remesher.
    anisotropy = min(flow.anisotropy, 4.0)
    density_u = base_density * importance / anisotropy
    density_v = base_density * importance * anisotropy

    reasons = []
    if curvature > 0.15:
        reasons.append("curvature")
    if deformation_importance > 0.0:
        reasons.append("deformation")
    if silhouette_importance > 0.0:
        reasons.append("silhouette")
    if anisotropy > 1.05:
        reasons.append("anisotropic-flow")
    if not reasons:
        reasons.append("uniform-baseline")

    return DensityHint(
        face_index=face_index,
        density_u=density_u,
        density_v=density_v,
        principal_direction=flow.principal_direction,
        reason="+".join(reasons),
    )
