"""Photometric stereo on controlled multi-light captures.

With at least four declared light directions a per-pixel least-squares fit
recovers the surface normal, the albedo and an ambient term. Pixels that break
the Lambertian model, such as self-shadowed or specular ones, show up in the fit
residual, which is what the reliability gate uses. Everything here is
deterministic; the geometry helpers exist so a result can be scored against the
normals that generated it.
"""

from __future__ import annotations

import math

Floats = list[list[float]]
Normals = list[list[tuple[float, float, float] | None]]

MIN_LIGHTS = 4
TINY = 1e-9


def light_directions(count: int = 6) -> list[tuple[float, float, float]]:
    """Unit vectors pointing towards the lights, spread over the upper hemisphere."""
    if isinstance(count, bool) or not isinstance(count, int) or count < MIN_LIGHTS:
        raise ValueError(f"at least {MIN_LIGHTS} light directions are needed")
    directions = []
    for index in range(count):
        azimuth = 2.0 * math.pi * index / count
        elevation = math.radians(35.0 + 20.0 * (index % 3))
        directions.append((
            round(math.cos(elevation) * math.cos(azimuth), 6),
            round(math.cos(elevation) * math.sin(azimuth), 6),
            round(math.sin(elevation), 6),
        ))
    return directions


def normalize(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    length = math.sqrt(sum(value * value for value in vector))
    if length <= TINY:
        raise ValueError("cannot normalize a zero vector")
    return (vector[0] / length, vector[1] / length, vector[2] / length)


def validate_lights(directions: object) -> list[tuple[float, float, float]]:
    if not isinstance(directions, (list, tuple)) or len(directions) < MIN_LIGHTS:
        raise ValueError(f"at least {MIN_LIGHTS} light directions are required")
    out = []
    for index, direction in enumerate(directions):
        if not isinstance(direction, (list, tuple)) or len(direction) != 3:
            raise ValueError(f"light {index} must have three components")
        values = []
        for value in direction:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"light {index} components must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError(f"light {index} components must be finite")
            values.append(float(value))
        out.append(normalize((values[0], values[1], values[2])))
    return out


def _solve(matrix: list[list[float]], vector: list[float]) -> list[float] | None:
    """Gaussian elimination with partial pivoting; None when the system is singular."""
    size = len(matrix)
    augmented = [row[:] + [vector[index]] for index, row in enumerate(matrix)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < TINY:
            return None
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        for row in range(column + 1, size):
            factor = augmented[row][column] / augmented[column][column]
            for index in range(column, size + 1):
                augmented[row][index] -= factor * augmented[column][index]
    solution = [0.0] * size
    for row in range(size - 1, -1, -1):
        total = augmented[row][size]
        for column in range(row + 1, size):
            total -= augmented[row][column] * solution[column]
        solution[row] = total / augmented[row][row]
    return solution


def synthetic_geometry(kind: str, size: int = 32, *, height: float = 0.6,
                       period: float = 12.0) -> tuple[Normals, Floats]:
    """Known normals and a validity mask for controlled surfaces."""
    if isinstance(size, bool) or not isinstance(size, int) or size < 8:
        raise ValueError("size must be an integer of at least eight")
    center = (size - 1) / 2.0
    normals: Normals = [[None] * size for _ in range(size)]
    mask = [[0.0] * size for _ in range(size)]
    radius = size * 0.45
    for r in range(size):
        for c in range(size):
            x = (c - center) / radius
            y = (r - center) / radius
            if kind == "sphere":
                squared = x * x + y * y
                if squared > 1.0:
                    continue
                z = math.sqrt(max(1.0 - squared, 0.0))
                normals[r][c] = (x, y, z)
            elif kind == "tilted-plane":
                normals[r][c] = normalize((-0.35, -0.2, 1.0))
            elif kind == "waves":
                step = 2.0 * math.pi / period
                dzdx = height * step * math.cos(step * c) * math.sin(step * r)
                dzdy = height * step * math.sin(step * c) * math.cos(step * r)
                normals[r][c] = normalize((-dzdx, -dzdy, 1.0))
            elif kind == "cone":
                length = math.hypot(x, y)
                if length > 1.0:
                    continue
                # z = 1 - r, so the surface gradient is (x/r, y/r, 1): the plain
                # (x, y, 1) form would describe a paraboloid instead.
                if length > 1e-6:
                    normals[r][c] = normalize((x / length, y / length, 1.0))
                else:
                    normals[r][c] = (0.0, 0.0, 1.0)
            else:
                raise ValueError(f"unknown geometry: {kind}")
            mask[r][c] = 1.0
    return normals, mask


def multilight_capture(
    albedo: Floats,
    normals: Normals,
    lights: list[tuple[float, float, float]],
    *,
    ambient: float = 0.05,
    specular: float = 0.0,
    specular_exponent: float = 24.0,
) -> list[Floats]:
    """Lambertian captures, optionally with an ambient term and a specular lobe."""
    directions = validate_lights(lights)
    height = len(albedo)
    width = len(albedo[0])
    for value in (ambient, specular):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("ambient and specular must be numeric")
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError("ambient and specular must be between 0 and 1")
    captures = []
    for direction in directions:
        image = [[0.0] * width for _ in range(height)]
        for r in range(height):
            for c in range(width):
                normal = normals[r][c]
                if normal is None:
                    image[r][c] = 0.0
                    continue
                cosine = max(
                    0.0,
                    normal[0] * direction[0]
                    + normal[1] * direction[1]
                    + normal[2] * direction[2],
                )
                value = albedo[r][c] * cosine + float(ambient)
                if specular > 0.0:
                    half = normalize((
                        direction[0],
                        direction[1],
                        direction[2] + 1.0,
                    ))
                    highlight = max(
                        0.0,
                        normal[0] * half[0] + normal[1] * half[1] + normal[2] * half[2],
                    )
                    value += float(specular) * highlight ** specular_exponent
                image[r][c] = min(max(value, 0.0), 1.0)
        captures.append(image)
    return captures


def photometric_normals(
    images: list[Floats],
    lights: list[tuple[float, float, float]],
    *,
    ambient_unknown: bool = True,
) -> dict:
    """Per-pixel least squares over the light directions.

    Solves for albedo times the normal, plus an ambient term when it is declared
    unknown, then splits the albedo back out. The residual is computed against the
    full model, ambient included.
    """
    directions = validate_lights(lights)
    if not isinstance(images, list) or len(images) != len(directions):
        raise ValueError("one image per light direction is required")
    height = len(images[0])
    width = len(images[0][0])
    for index, image in enumerate(images):
        if len(image) != height or any(len(row) != width for row in image):
            raise ValueError(f"image {index} does not share the capture shape")
        for row in image:
            for value in row:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError("capture intensities must be numeric")
                if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                    raise ValueError("capture intensities must be between 0 and 1")
    size = 4 if ambient_unknown else 3
    normals: Normals = [[None] * width for _ in range(height)]
    albedo = [[0.0] * width for _ in range(height)]
    ambient_map = [[0.0] * width for _ in range(height)]
    residual = [[1.0] * width for _ in range(height)]
    for r in range(height):
        for c in range(width):
            matrix = [[0.0] * size for _ in range(size)]
            vector = [0.0] * size
            for index, direction in enumerate(directions):
                row = [direction[0], direction[1], direction[2]]
                if ambient_unknown:
                    row.append(1.0)
                intensity = float(images[index][r][c])
                for a in range(size):
                    for b in range(size):
                        matrix[a][b] += row[a] * row[b]
                    vector[a] += row[a] * intensity
            solution = _solve(matrix, vector)
            if solution is None:
                continue
            scaled = (solution[0], solution[1], solution[2])
            length = math.sqrt(sum(value * value for value in scaled))
            if length <= 1e-6:
                continue
            normals[r][c] = (scaled[0] / length, scaled[1] / length, scaled[2] / length)
            albedo[r][c] = min(length, 1.0)
            if ambient_unknown:
                ambient_map[r][c] = solution[3]
            total = 0.0
            for index, direction in enumerate(directions):
                cosine = max(
                    0.0,
                    normals[r][c][0] * direction[0]
                    + normals[r][c][1] * direction[1]
                    + normals[r][c][2] * direction[2],
                )
                predicted = min(
                    max(albedo[r][c] * cosine + ambient_map[r][c], 0.0), 1.0
                )
                total += (predicted - float(images[index][r][c])) ** 2
            residual[r][c] = min(math.sqrt(total / len(directions)), 1.0)
    return {
        "normals": normals,
        "albedo": albedo,
        "ambient": ambient_map,
        "residual": residual,
        "method": "photometric-stereo-least-squares",
        "lights": len(directions),
        "ambient_unknown": bool(ambient_unknown),
    }


def normal_error(recovered: Normals, truth: Normals, mask: Floats | None = None,
                 *, indices: list[tuple[int, int]] | None = None) -> dict:
    """Angular error in degrees over the valid pixels, or over a given subset."""
    errors = []
    pairs = indices if indices is not None else [
        (r, c)
        for r in range(len(truth))
        for c in range(len(truth[0]))
        if truth[r][c] is not None and (mask is None or mask[r][c] > 0.0)
    ]
    missing = 0
    for r, c in pairs:
        truth_normal = truth[r][c]
        estimate = recovered[r][c] if r < len(recovered) and c < len(recovered[0]) else None
        if truth_normal is None:
            continue
        if estimate is None:
            missing += 1
            continue
        dot = sum(
            truth_normal[index] * estimate[index] for index in range(3)
        )
        errors.append(math.degrees(math.acos(min(max(dot, -1.0), 1.0))))
    if not errors:
        return {"pixels": 0, "mean_deg": None, "p95_deg": None, "max_deg": None,
                "missing": missing}
    ordered = sorted(errors)
    return {
        "pixels": len(errors),
        "mean_deg": round(sum(errors) / len(errors), 4),
        "p95_deg": round(ordered[int(0.95 * (len(ordered) - 1))], 4),
        "max_deg": round(ordered[-1], 4),
        "missing": missing,
    }


def reliability_features(
    residual: Floats,
    captures: list[Floats] | None = None,
    *,
    patch: int = 8,
) -> list[dict]:
    """Per-patch statistics behind the trust decision, all in [0, 1]."""
    if isinstance(patch, bool) or not isinstance(patch, int) or patch < 2:
        raise ValueError("patch must be an integer of at least two")
    height = len(residual)
    width = len(residual[0])
    patches = []
    for row_start in range(0, height - patch + 1, patch):
        for column_start in range(0, width - patch + 1, patch):
            values = [
                residual[r][c]
                for r in range(row_start, row_start + patch)
                for c in range(column_start, column_start + patch)
            ]
            count = len(values)
            mean = sum(values) / count
            variance = sum((value - mean) ** 2 for value in values) / count
            gradient = 0.0
            for r in range(row_start, row_start + patch - 1):
                for c in range(column_start, column_start + patch - 1):
                    gradient += abs(residual[r][c + 1] - residual[r][c])
                    gradient += abs(residual[r + 1][c] - residual[r][c])
            features = {
                "mean_residual": round(min(mean, 1.0), 6),
                "max_residual": round(min(max(values), 1.0), 6),
                "residual_std": round(min(math.sqrt(variance) * 2.0, 1.0), 6),
                "residual_gradient": round(min(gradient / (2 * count), 1.0), 6),
            }
            if captures:
                intensities = [
                    float(image[r][c])
                    for image in captures
                    for r in range(row_start, row_start + patch)
                    for c in range(column_start, column_start + patch)
                ]
                features["mean_intensity"] = round(
                    min(sum(intensities) / len(intensities), 1.0), 6
                )
                features["intensity_range"] = round(
                    min(max(intensities) - min(intensities), 1.0), 6
                )
            patches.append({
                "row": row_start,
                "column": column_start,
                "features": features,
                "indices": [
                    (r, c)
                    for r in range(row_start, row_start + patch)
                    for c in range(column_start, column_start + patch)
                ],
            })
    return patches


def reliable_patches(patches: list[dict], *, residual_limit: float = 0.02) -> list[dict]:
    """Published rule: a patch is trusted when its mean residual stays low."""
    if isinstance(residual_limit, bool) or not isinstance(residual_limit, (int, float)):
        raise ValueError("residual_limit must be numeric")
    if not 0.0 < float(residual_limit) < 1.0:
        raise ValueError("residual_limit must be in (0, 1)")
    trusted = []
    for patch in patches:
        entry = dict(patch)
        entry["trusted"] = patch["features"]["mean_residual"] <= float(residual_limit)
        trusted.append(entry)
    return trusted
