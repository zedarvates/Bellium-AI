"""Consultative quad-strip proposals between two retopology guide paths.

The output references existing mesh vertices only. No source topology is changed.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from bellium.geometry.mesh import Mesh, Vec3
from bellium.geometry.retopo import RetopoPath

Quad = tuple[int, int, int, int]


@dataclass(frozen=True)
class QuadStripProposal:
    quads: tuple[Quad, ...]
    left_vertices: tuple[int, ...]
    right_vertices: tuple[int, ...]
    score: float
    reason: str
    rejected: bool
    rejection_reason: str | None = None


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


def path_vertices(path: RetopoPath) -> tuple[int, ...]:
    if not path.edges:
        return ()
    vertices = [path.edges[0][0], path.edges[0][1]]
    for edge in path.edges[1:]:
        if edge[0] == vertices[-1]:
            vertices.append(edge[1])
        elif edge[1] == vertices[-1]:
            vertices.append(edge[0])
        else:
            raise ValueError("retopo path edges are not contiguous")
    return tuple(vertices)


def _segment_alignment(mesh: Mesh, left: tuple[int, ...], right: tuple[int, ...]) -> float:
    values: list[float] = []
    for index in range(len(left) - 1):
        left_dir = _normalize(_sub(mesh.vertices[left[index + 1]], mesh.vertices[left[index]]))
        right_dir = _normalize(_sub(mesh.vertices[right[index + 1]], mesh.vertices[right[index]]))
        values.append(abs(_dot(left_dir, right_dir)))
    return sum(values) / len(values) if values else 0.0


def _width_ratio(mesh: Mesh, left: tuple[int, ...], right: tuple[int, ...]) -> float:
    rung_lengths = [
        _norm(_sub(mesh.vertices[right[index]], mesh.vertices[left[index]]))
        for index in range(len(left))
    ]
    rail_lengths: list[float] = []
    for chain in (left, right):
        rail_lengths.extend(
            _norm(_sub(mesh.vertices[chain[index + 1]], mesh.vertices[chain[index]]))
            for index in range(len(chain) - 1)
        )
    mean_rung = sum(rung_lengths) / len(rung_lengths)
    mean_rail = sum(rail_lengths) / len(rail_lengths) if rail_lengths else 0.0
    if mean_rail <= 1e-12:
        return math.inf
    return mean_rung / mean_rail


def propose_quad_strip(
    mesh: Mesh,
    left_path: RetopoPath,
    right_path: RetopoPath,
    *,
    min_alignment: float = 0.75,
    max_width_ratio: float = 4.0,
) -> QuadStripProposal:
    if not 0.0 <= min_alignment <= 1.0:
        raise ValueError("min_alignment must be between 0 and 1")
    if max_width_ratio <= 0.0:
        raise ValueError("max_width_ratio must be > 0")

    if left_path.ambiguous or right_path.ambiguous:
        return QuadStripProposal((), (), (), 0.0, "quad-strip", True, "ambiguous_path")
    if left_path.closed != right_path.closed:
        return QuadStripProposal((), (), (), 0.0, "quad-strip", True, "closure_mismatch")

    try:
        left = path_vertices(left_path)
        right = path_vertices(right_path)
    except ValueError:
        return QuadStripProposal((), (), (), 0.0, "quad-strip", True, "non_contiguous_path")

    if len(left) != len(right):
        return QuadStripProposal((), left, right, 0.0, "quad-strip", True, "vertex_count_mismatch")
    if len(left) < 2:
        return QuadStripProposal((), left, right, 0.0, "quad-strip", True, "path_too_short")
    if set(left) & set(right):
        return QuadStripProposal((), left, right, 0.0, "quad-strip", True, "paths_share_vertices")

    alignment = _segment_alignment(mesh, left, right)
    if alignment < min_alignment:
        return QuadStripProposal(
            (), left, right, alignment, "quad-strip", True, "direction_mismatch"
        )

    width_ratio = _width_ratio(mesh, left, right)
    if not math.isfinite(width_ratio) or width_ratio > max_width_ratio:
        return QuadStripProposal(
            (), left, right, alignment, "quad-strip", True, "strip_too_wide"
        )

    quads = tuple(
        (left[index], left[index + 1], right[index + 1], right[index])
        for index in range(len(left) - 1)
    )
    score = alignment * (1.0 / (1.0 + 0.25 * width_ratio))
    return QuadStripProposal(
        quads=quads,
        left_vertices=left,
        right_vertices=right,
        score=max(0.0, min(1.0, score)),
        reason="quad-strip",
        rejected=False,
    )
