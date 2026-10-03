"""Execute the safe subset of primitive-mesh-plan-v0 into a simple mesh scene."""

from __future__ import annotations

from dataclasses import dataclass

from bellium.geometry.mesh import Mesh, box, combine, cylinder, mirror, sphere, transform
from bellium.geometry.plan import ConstructionPlan


@dataclass(frozen=True)
class ExecutionResult:
    meshes: dict[str, Mesh]
    combined: Mesh
    executed_operations: tuple[str, ...]
    skipped_operations: tuple[str, ...]


def _part_mesh(part) -> Mesh:
    dims = part.dimensions
    if part.primitive == "box":
        if len(dims) != 3:
            raise ValueError(f"{part.id}: box needs 3 dimensions")
        mesh = box(*dims)
    elif part.primitive == "cylinder":
        if len(dims) != 2:
            raise ValueError(f"{part.id}: cylinder needs 2 dimensions")
        mesh = cylinder(radius=dims[0] / 2.0, length=dims[1])
    elif part.primitive == "sphere":
        if len(dims) != 1:
            raise ValueError(f"{part.id}: sphere needs 1 dimension")
        mesh = sphere(radius=dims[0] / 2.0)
    else:
        raise ValueError(f"{part.id}: primitive execution not implemented: {part.primitive}")
    return transform(
        mesh,
        part.transform["translation"],
        part.transform["rotation_euler"],
        part.transform["scale"],
    )


def execute_plan(plan: ConstructionPlan) -> ExecutionResult:
    meshes = {part.id: _part_mesh(part) for part in plan.parts}
    executed: list[str] = []
    skipped: list[str] = []

    for operation in plan.operations:
        if operation.operator == "mirror":
            source = meshes[operation.input_ids[0]]
            axis = operation.parameters.get("axis", "x")
            offset = float(operation.parameters.get("offset", 0.0))
            mirrored = mirror(source, axis=axis, offset=offset)
            target = operation.output_ids[0] if operation.output_ids else operation.id
            meshes[target] = combine((source, mirrored))
            executed.append(operation.id)
        elif operation.operator == "bevel":
            # Geometry bevel is intentionally not implemented yet. Preserve a
            # deterministic marker while keeping the source mesh unchanged.
            source = meshes[operation.input_ids[0]]
            target = operation.output_ids[0] if operation.output_ids else operation.id
            meshes[target] = Mesh(
                vertices=source.vertices,
                faces=source.faces,
                metadata=source.metadata + (("bevel-pending", operation.id),),
            )
            skipped.append(operation.id)
        else:
            skipped.append(operation.id)

    return ExecutionResult(
        meshes=meshes,
        combined=combine(meshes.values()),
        executed_operations=tuple(executed),
        skipped_operations=tuple(skipped),
    )
