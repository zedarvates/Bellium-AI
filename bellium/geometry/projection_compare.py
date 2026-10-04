"""Compare cheap neighbour projection against exact triangle projection."""

from __future__ import annotations

from dataclasses import dataclass
import math

from bellium.geometry.closest import closest_point_on_mesh
from bellium.geometry.mesh import Mesh, Vec3
from bellium.geometry.projection import project_to_neighbour_centroid


@dataclass(frozen=True)
class ProjectionComparison:
    query: Vec3
    neighbour_point: Vec3 | None
    exact_point: Vec3 | None
    point_error: float | None
    exact_distance: float | None
    neighbour_confidence: float
    recommendation: str
    abstained: bool


def _distance(a: Vec3, b: Vec3) -> float:
    dx = a[0]-b[0]
    dy = a[1]-b[1]
    dz = a[2]-b[2]
    return math.sqrt(dx*dx + dy*dy + dz*dz)


def compare_projection(
    mesh: Mesh,
    query: Vec3,
    *,
    k: int = 4,
    max_radius: float | None = None,
    tolerance: float = 0.02,
) -> ProjectionComparison:
    if tolerance < 0.0:
        raise ValueError("tolerance must be >= 0")

    neighbour = project_to_neighbour_centroid(mesh, query, k=k, max_radius=max_radius)
    exact = closest_point_on_mesh(mesh, query)

    if neighbour.abstained or exact.abstained:
        return ProjectionComparison(
            query=query,
            neighbour_point=None if neighbour.abstained else neighbour.point,
            exact_point=None if exact.abstained else exact.point,
            point_error=None,
            exact_distance=None if exact.abstained else exact.distance,
            neighbour_confidence=neighbour.confidence,
            recommendation="exact-or-abstain",
            abstained=True,
        )

    error = _distance(neighbour.point, exact.point)
    recommendation = "neighbour-ok" if error <= tolerance else "use-exact"
    return ProjectionComparison(
        query=query,
        neighbour_point=neighbour.point,
        exact_point=exact.point,
        point_error=error,
        exact_distance=exact.distance,
        neighbour_confidence=neighbour.confidence,
        recommendation=recommendation,
        abstained=False,
    )
