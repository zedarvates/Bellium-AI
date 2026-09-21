from __future__ import annotations

from pathlib import Path
from typing import Any

from bellium.knn._image import shape
from bellium.magick.auto_orient import auto_orient
from bellium.magick.fastimage import fast_info
from bellium.magick.geometry_parser import parse_geometry
from bellium.magick.io import load_image, save_image
from bellium.specialists.magick import magick_process

def run_pipeline_on_image(
    image: list[list[tuple[int, int, int]]],
    operations: list[dict[str, Any]],
) -> list[list[tuple[int, int, int]]]:
    current = image
    for op in operations:
        op_name = op.get('operation')
        query = dict(op)
        query['image'] = current
        if op_name == 'resize' and 'geometry' in op:
            in_h, in_w = shape(current)
            tw, th, _ = parse_geometry(op['geometry'], in_w, in_h)
            query['width'] = tw
            query['height'] = th
        res = magick_process(query)
        if res.abstained:
            raise RuntimeError(f'operation {op_name} abstained: {res.output.get("reason")}')
        current = res.output['image']
    return current

def mogrify_batch(
    files: list[Path],
    operations: list[dict[str, Any]],
    *,
    output_dir: Path | None = None,
    output_format: str | None = None,
    do_auto_orient: bool = False,
) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        try:
            img = load_image(f)
            if do_auto_orient:
                fi = fast_info(f)
                img = auto_orient(img, fi.orientation)
            processed = run_pipeline_on_image(img, operations)
            dest_fmt = output_format if output_format else f.suffix.lstrip('.')
            if output_dir is not None:
                dest = output_dir / f'{f.stem}.{dest_fmt}'
            else:
                dest = f.with_suffix(f'.{dest_fmt}')
            save_image(processed, dest)
            h, w = shape(processed)
            reports.append({
                'input': str(f),
                'output': str(dest),
                'status': 'ok',
                'dimensions': f'{w}x{h}',
            })
        except Exception as e:
            reports.append({
                'input': str(f),
                'status': 'error',
                'error': str(e),
            })
    return reports
