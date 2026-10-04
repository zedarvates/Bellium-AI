"""Nearest-neighbour surface projection helpers.

This intentionally starts with a transparent inverse-distance weighted centroid.
It is useful for pulling newly proposed retopology vertices back toward a dense
reference surface, but it is not a substitute for exact closest-point-on-triangle
projection.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from bellium.geometry.mesh import Mesh, Vec3


@dataclass(frozen=True)
class ProjectionResult:
    point: Vec3
    source_indices: tuple[int, ...]
    distances: tuple[float, ...]
    confidence: float
    abstained: bool
    reason: str


def _distance(a: Vec3, b: Vec3) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return math.sqrt(dx*dx + dy*dy + dz*dz)


def nearest_vertices(mesh: Mesh, point: Vec3, k: int = 4) -> tuple[tuple[int, float], ...]:
    if k < 1:
        raise ValueError("k must be >= 1")
    if not mesh.vertices:
        return ()
    ranked = sorted(
        ((index, _distance(vertex, point)) for index, vertex in enumerate(mesh.vertices)),
        key=lambda item: (item[1], item[0]),
    )
    return tuple(ranked[: min(k, len(ranked))])


def project_to_neighbour_centroid(
    mesh: Mesh,
    point: Vec3,
    *,
    k: int = 4,
    max_radius: float | None = None,
) -> ProjectionResult:
    if max_radius is not None and max_radius <= 0.0:
        raise ValueError("max_radius must be > 0")

    neighbours = nearest_vertices(mesh, point, k=k)
    if not neighbours:
        return ProjectionResult(point, (), (), 0.0, True, "empty_reference_mesh")

    if neighbours[0][1] <= 1e-12:
        index = neighbours[0][0]
        return ProjectionResult(
            mesh.vertices[index], (index,), (0.0,), 1.0, False, "exact_vertex"
        )

    if max_radius is not None and neighbours[0][1] > max_radius:
        return ProjectionResult(
            point,
            tuple(index for index, _ in neighbours),
            tuple(distance for _, distance in neighbours),
            0.0,
            True,
            "outside_projection_radius",
        )

    selected = [
        (index, distance)
        for index, distance in neighbours
        if max_radius is None or distance <= max_radius
    ]
    if not selected:
        return ProjectionResult(point, (), (), 0.0, True, "no_neighbour_in_radius")

    weights = [1.0 / max(distance, 1e-12) for _, distance in selected]
    total = sum(weights)
    x = sum(mesh.vertices[index][0] * weight for (index, _), weight in zip(selected, weights))
    y = sum(mesh.vertices[index][1] * weight for (index, _), weight in zip(selected, weights))
    z = sum(mesh.vertices[index][2] * weight for (index, _), weight in zip(selected, weights))
    projected = (x / total, y / total, z / total)

    mean_distance = sum(distance for _, distance in selected) / len(selected)
    spread = max(distance for _, distance in selected) - min(distance for _, distance in selected)
    confidence = 1.0 / (1.0 + mean_distance + spread)

    return ProjectionResult(
        point=projected,
        source_indices=tuple(index for index, _ in selected),
        distances=tuple(distance for _, distance in selected),
        confidence=max(0.0, min(1.0, confidence)),
        abstained=False,
        reason="inverse-distance-neighbour-centroid",
    )
