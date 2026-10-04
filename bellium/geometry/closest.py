"""Exact closest-point helpers for triangle reference projection.

This provides a deterministic reference against which cheaper neighbour-based
projection can be measured. Polygon faces are triangulated as a fan for now.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from bellium.geometry.mesh import Mesh, Vec3


@dataclass(frozen=True)
class ClosestPointResult:
    point: Vec3
    distance: float
    face_index: int | None
    triangle: tuple[int, int, int] | None
    abstained: bool
    reason: str


def _sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0]-b[0], a[1]-b[1], a[2]-b[2])


def _add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0]+b[0], a[1]+b[1], a[2]+b[2])


def _scale(v: Vec3, s: float) -> Vec3:
    return (v[0]*s, v[1]*s, v[2]*s)


def _dot(a: Vec3, b: Vec3) -> float:
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]


def _distance(a: Vec3, b: Vec3) -> float:
    d = _sub(a, b)
    return math.sqrt(_dot(d, d))


def closest_point_on_triangle(point: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Vec3:
    """Real-Time Collision Detection style barycentric region test."""
    ab = _sub(b, a)
    ac = _sub(c, a)
    ap = _sub(point, a)
    d1 = _dot(ab, ap)
    d2 = _dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return a

    bp = _sub(point, b)
    d3 = _dot(ab, bp)
    d4 = _dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        return b

    vc = d1*d4 - d3*d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return _add(a, _scale(ab, v))

    cp = _sub(point, c)
    d5 = _dot(ab, cp)
    d6 = _dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        return c

    vb = d5*d2 - d1*d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return _add(a, _scale(ac, w))

    va = d3*d6 - d5*d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        bc = _sub(c, b)
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return _add(b, _scale(bc, w))

    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    return _add(a, _add(_scale(ab, v), _scale(ac, w)))


def closest_point_on_mesh(mesh: Mesh, point: Vec3) -> ClosestPointResult:
    best: tuple[float, Vec3, int, tuple[int, int, int]] | None = None
    for face_index, face in enumerate(mesh.faces):
        if len(face) < 3:
            continue
        anchor = face[0]
        for i in range(1, len(face)-1):
            tri = (anchor, face[i], face[i+1])
            a, b, c = (mesh.vertices[index] for index in tri)
            candidate = closest_point_on_triangle(point, a, b, c)
            distance = _distance(point, candidate)
            current = (distance, candidate, face_index, tri)
            if best is None or (distance, face_index, tri) < (best[0], best[2], best[3]):
                best = current
    if best is None:
        return ClosestPointResult(point, math.inf, None, None, True, "no_valid_triangles")
    return ClosestPointResult(
        point=best[1],
        distance=best[0],
        face_index=best[2],
        triangle=best[3],
        abstained=False,
        reason="closest-triangle",
    )
