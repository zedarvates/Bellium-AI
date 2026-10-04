"""Small deterministic exporters and measurements for Bellium geometry."""

from __future__ import annotations

from bellium.geometry.mesh import Mesh


def bounds(mesh: Mesh) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if not mesh.vertices:
        raise ValueError("mesh has no vertices")
    xs = [v[0] for v in mesh.vertices]
    ys = [v[1] for v in mesh.vertices]
    zs = [v[2] for v in mesh.vertices]
    return ((min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs)))


def extent(mesh: Mesh) -> tuple[float, float, float]:
    low, high = bounds(mesh)
    return (high[0] - low[0], high[1] - low[1], high[2] - low[2])


def mesh_to_obj(mesh: Mesh, object_name: str = "bellium") -> str:
    lines = [f"o {object_name}"]
    for x, y, z in mesh.vertices:
        lines.append(f"v {x:.9g} {y:.9g} {z:.9g}")
    for face in mesh.faces:
        if len(face) < 3:
            raise ValueError("OBJ faces need at least three vertices")
        lines.append("f " + " ".join(str(index + 1) for index in face))
    return "\n".join(lines) + "\n"
