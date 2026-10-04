"""Small bounded Quadric Error Function solver for feature-preserving placement."""

from __future__ import annotations

from dataclasses import dataclass
import math

from bellium.geometry.mesh import Vec3


@dataclass(frozen=True)
class QEFConstraint:
    point: Vec3
    normal: Vec3


@dataclass(frozen=True)
class QEFResult:
    point: Vec3 | None
    error: float | None
    abstained: bool
    reason: str


def _dot(a: Vec3, b: Vec3) -> float:
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]


def _norm(v: Vec3) -> float:
    return math.sqrt(_dot(v, v))


def _normalize(v: Vec3) -> Vec3:
    n = _norm(v)
    if n <= 1e-12:
        raise ValueError("QEF normal must be non-zero")
    return (v[0]/n, v[1]/n, v[2]/n)


def _solve3(matrix: list[list[float]], rhs: list[float]) -> tuple[float, float, float] | None:
    a = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(3):
        pivot = max(range(col, 3), key=lambda row: abs(a[row][col]))
        if abs(a[pivot][col]) <= 1e-10:
            return None
        if pivot != col:
            a[col], a[pivot] = a[pivot], a[col]
        value = a[col][col]
        for j in range(col, 4):
            a[col][j] /= value
        for row in range(3):
            if row == col:
                continue
            factor = a[row][col]
            for j in range(col, 4):
                a[row][j] -= factor * a[col][j]
    return (a[0][3], a[1][3], a[2][3])


def qef_error(point: Vec3, constraints: tuple[QEFConstraint, ...]) -> float:
    total = 0.0
    for constraint in constraints:
        normal = _normalize(constraint.normal)
        residual = _dot(normal, point) - _dot(normal, constraint.point)
        total += residual * residual
    return total


def solve_qef(
    constraints: tuple[QEFConstraint, ...],
    *,
    bounds: tuple[Vec3, Vec3] | None = None,
) -> QEFResult:
    if len(constraints) < 3:
        return QEFResult(None, None, True, "insufficient_constraints")

    ata = [[0.0 for _ in range(3)] for _ in range(3)]
    atb = [0.0, 0.0, 0.0]
    normalized: list[QEFConstraint] = []

    for constraint in constraints:
        normal = _normalize(constraint.normal)
        normalized.append(QEFConstraint(constraint.point, normal))
        b = _dot(normal, constraint.point)
        for i in range(3):
            atb[i] += normal[i] * b
            for j in range(3):
                ata[i][j] += normal[i] * normal[j]

    solved = _solve3(ata, atb)
    if solved is None:
        return QEFResult(None, None, True, "singular_constraints")

    point: Vec3 = solved
    if bounds is not None:
        low, high = bounds
        point = (
            min(high[0], max(low[0], point[0])),
            min(high[1], max(low[1], point[1])),
            min(high[2], max(low[2], point[2])),
        )

    normalized_tuple = tuple(normalized)
    return QEFResult(point, qef_error(point, normalized_tuple), False, "solved")
