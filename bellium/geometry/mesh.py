"""Minimal deterministic mesh kernel for ShapeGrammar 3D experiments.

This is intentionally small: authored primitives, affine transforms, mirror and
simple bevel metadata. It is not a production CAD or retopology kernel.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

Vec3 = tuple[float, float, float]
Face = tuple[int, ...]


@dataclass(frozen=True)
class Mesh:
    vertices: tuple[Vec3, ...]
    faces: tuple[Face, ...]
    metadata: tuple[tuple[str, str], ...] = ()


def box(width: float, height: float, depth: float) -> Mesh:
    hx, hy, hz = width / 2.0, height / 2.0, depth / 2.0
    vertices = (
        (-hx, -hy, -hz), (hx, -hy, -hz), (hx, hy, -hz), (-hx, hy, -hz),
        (-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz),
    )
    faces = (
        (0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1),
        (1, 5, 6, 2), (2, 6, 7, 3), (4, 0, 3, 7),
    )
    return Mesh(vertices=vertices, faces=faces, metadata=(("primitive", "box"),))


def cylinder(radius: float, length: float, segments: int = 16) -> Mesh:
    if segments < 3:
        raise ValueError("segments must be >= 3")
    if radius <= 0.0 or length <= 0.0:
        raise ValueError("cylinder dimensions must be > 0")
    half = length / 2.0
    ring0: list[Vec3] = []
    ring1: list[Vec3] = []
    for i in range(segments):
        angle = 2.0 * math.pi * i / segments
        x = radius * math.cos(angle)
        z = radius * math.sin(angle)
        ring0.append((x, -half, z))
        ring1.append((x, half, z))
    vertices = tuple(ring0 + ring1 + [(0.0, -half, 0.0), (0.0, half, 0.0)])
    bottom_center = segments * 2
    top_center = bottom_center + 1
    faces: list[Face] = []
    for i in range(segments):
        nxt = (i + 1) % segments
        faces.append((i, nxt, segments + nxt, segments + i))
        faces.append((bottom_center, nxt, i))
        faces.append((top_center, segments + i, segments + nxt))
    return Mesh(vertices=vertices, faces=tuple(faces), metadata=(("primitive", "cylinder"),))


def sphere(radius: float, rings: int = 8, segments: int = 16) -> Mesh:
    if radius <= 0.0:
        raise ValueError("radius must be > 0")
    if rings < 2 or segments < 3:
        raise ValueError("sphere resolution too low")
    vertices: list[Vec3] = [(0.0, radius, 0.0)]
    for r in range(1, rings):
        phi = math.pi * r / rings
        y = radius * math.cos(phi)
        rr = radius * math.sin(phi)
        for s in range(segments):
            theta = 2.0 * math.pi * s / segments
            vertices.append((rr * math.cos(theta), y, rr * math.sin(theta)))
    vertices.append((0.0, -radius, 0.0))
    north = 0
    south = len(vertices) - 1
    faces: list[Face] = []
    for s in range(segments):
        nxt = (s + 1) % segments
        faces.append((north, 1 + s, 1 + nxt))
    for r in range(rings - 2):
        base = 1 + r * segments
        next_base = base + segments
        for s in range(segments):
            nxt = (s + 1) % segments
            faces.append((base + s, next_base + s, next_base + nxt, base + nxt))
    last_ring = 1 + (rings - 2) * segments
    for s in range(segments):
        nxt = (s + 1) % segments
        faces.append((last_ring + s, south, last_ring + nxt))
    return Mesh(vertices=tuple(vertices), faces=tuple(faces), metadata=(("primitive", "sphere"),))


def _rotate_xyz(v: Vec3, degrees: Vec3) -> Vec3:
    x, y, z = v
    rx, ry, rz = (math.radians(item) for item in degrees)
    cy, sy = math.cos(rx), math.sin(rx)
    y, z = y * cy - z * sy, y * sy + z * cy
    cy, sy = math.cos(ry), math.sin(ry)
    x, z = x * cy + z * sy, -x * sy + z * cy
    cy, sy = math.cos(rz), math.sin(rz)
    x, y = x * cy - y * sy, x * sy + y * cy
    return (x, y, z)


def transform(mesh: Mesh, translation: Vec3, rotation_euler: Vec3, scale: Vec3) -> Mesh:
    vertices: list[Vec3] = []
    for x, y, z in mesh.vertices:
        rotated = _rotate_xyz((x * scale[0], y * scale[1], z * scale[2]), rotation_euler)
        vertices.append((
            rotated[0] + translation[0],
            rotated[1] + translation[1],
            rotated[2] + translation[2],
        ))
    return Mesh(vertices=tuple(vertices), faces=mesh.faces, metadata=mesh.metadata)


def mirror(mesh: Mesh, axis: str, offset: float = 0.0) -> Mesh:
    if axis not in {"x", "y", "z"}:
        raise ValueError("mirror axis must be x, y or z")
    index = {"x": 0, "y": 1, "z": 2}[axis]
    mirrored: list[Vec3] = []
    for vertex in mesh.vertices:
        values = list(vertex)
        values[index] = 2.0 * offset - values[index]
        mirrored.append(tuple(values))  # type: ignore[arg-type]
    faces = tuple(tuple(reversed(face)) for face in mesh.faces)
    return Mesh(vertices=tuple(mirrored), faces=faces, metadata=mesh.metadata + (("mirror", axis),))


def combine(meshes: Iterable[Mesh]) -> Mesh:
    vertices: list[Vec3] = []
    faces: list[Face] = []
    metadata: list[tuple[str, str]] = []
    offset = 0
    for mesh in meshes:
        vertices.extend(mesh.vertices)
        faces.extend(tuple(index + offset for index in face) for face in mesh.faces)
        metadata.extend(mesh.metadata)
        offset += len(mesh.vertices)
    return Mesh(vertices=tuple(vertices), faces=tuple(faces), metadata=tuple(metadata))
