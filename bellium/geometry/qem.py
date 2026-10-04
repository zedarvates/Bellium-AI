"""Minimal QEM primitives for bounded simplification experiments.

This module evaluates candidate edge-collapse points. It does not perform a full
mesh simplification pass yet.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from bellium.geometry.mesh import Vec3

Quadric = tuple[tuple[float, float, float, float], ...]


@dataclass(frozen=True)
class CollapseCandidate:
    point: Vec3
    cost: float
    protected: bool
    reason: str


def plane_quadric(point: Vec3, normal: Vec3) -> Quadric:
    length = math.sqrt(normal[0]**2 + normal[1]**2 + normal[2]**2)
    if length <= 1e-12:
        raise ValueError("plane normal must be non-zero")
    a, b, c = (normal[0]/length, normal[1]/length, normal[2]/length)
    d = -(a*point[0] + b*point[1] + c*point[2])
    v = (a, b, c, d)
    return tuple(
        tuple(v[i] * v[j] for j in range(4))
        for i in range(4)
    )


def add_quadrics(left: Quadric, right: Quadric) -> Quadric:
    return tuple(
        tuple(left[i][j] + right[i][j] for j in range(4))
        for i in range(4)
    )


def quadric_error(quadric: Quadric, point: Vec3) -> float:
    v = (point[0], point[1], point[2], 1.0)
    total = 0.0
    for i in range(4):
        for j in range(4):
            total += v[i] * quadric[i][j] * v[j]
    return max(0.0, total)


def best_edge_collapse_candidate(
    left_point: Vec3,
    right_point: Vec3,
    left_quadric: Quadric,
    right_quadric: Quadric,
    *,
    protected: bool = False,
) -> CollapseCandidate:
    if protected:
        midpoint = (
            (left_point[0] + right_point[0]) / 2.0,
            (left_point[1] + right_point[1]) / 2.0,
            (left_point[2] + right_point[2]) / 2.0,
        )
        return CollapseCandidate(midpoint, math.inf, True, "protected_feature")

    quadric = add_quadrics(left_quadric, right_quadric)
    midpoint = (
        (left_point[0] + right_point[0]) / 2.0,
        (left_point[1] + right_point[1]) / 2.0,
        (left_point[2] + right_point[2]) / 2.0,
    )
    candidates = (left_point, right_point, midpoint)
    ranked = sorted(
        ((quadric_error(quadric, point), index, point) for index, point in enumerate(candidates)),
        key=lambda item: (item[0], item[1]),
    )
    cost, _, point = ranked[0]
    return CollapseCandidate(point, cost, False, "lowest_qem_cost")
