from __future__ import annotations

import json
from pathlib import Path

from bellium.knn._image import Image, Rgb, clamp_rgb, shape

def load_ppm(data: bytes) -> Image:
    idx = 0
    header_tokens: list[bytes] = []
    while len(header_tokens) < 4 and idx < len(data):
        while idx < len(data) and data[idx] in b' \t\r\n':
            idx += 1
        if idx >= len(data):
            break
        if data[idx] == ord(b'#'):
            while idx < len(data) and data[idx] != ord(b'\n'):
                idx += 1
            continue
        token_start = idx
        while idx < len(data) and data[idx] not in b' \t\r\n#':
            idx += 1
        header_tokens.append(data[token_start:idx])
    if len(header_tokens) < 4:
        raise ValueError('truncated PPM header')
    magic, w_bytes, h_bytes, maxval_bytes = header_tokens[:4]
    width = int(w_bytes)
    height = int(h_bytes)
    maxval = int(maxval_bytes)
    if width <= 0 or height <= 0 or maxval <= 0:
        raise ValueError(f'invalid PPM dimensions or maxval: {width}x{height}, maxval {maxval}')
    if idx < len(data) and data[idx] == ord(b'\r'):
        idx += 1
    if idx < len(data) and data[idx] in b' \t\n':
        idx += 1
    pixels: list[list[Rgb]] = []
    if magic == b'P6':
        payload = data[idx:]
        expected_bytes = width * height * 3
        if len(payload) < expected_bytes:
            raise ValueError(f'truncated P6 data: got {len(payload)} bytes, expected {expected_bytes}')
        p_idx = 0
        scale = 255.0 / maxval
        for r in range(height):
            row = []
            for c in range(width):
                r_val = clamp_rgb(payload[p_idx] * scale)
                g_val = clamp_rgb(payload[p_idx + 1] * scale)
                b_val = clamp_rgb(payload[p_idx + 2] * scale)
                row.append((r_val, g_val, b_val))
                p_idx += 3
            pixels.append(row)
    elif magic == b'P3':
        numbers = [int(tok) for tok in data[idx:].split()]
        if len(numbers) < width * height * 3:
            raise ValueError('truncated P3 data')
        scale = 255.0 / maxval
        n_idx = 0
        for r in range(height):
            row = []
            for c in range(width):
                r_val = clamp_rgb(numbers[n_idx] * scale)
                g_val = clamp_rgb(numbers[n_idx + 1] * scale)
                b_val = clamp_rgb(numbers[n_idx + 2] * scale)
                row.append((r_val, g_val, b_val))
                n_idx += 3
            pixels.append(row)
    else:
        raise ValueError(f'unsupported PPM magic: {magic.decode("ascii", "replace")}')
    return pixels

def save_ppm(image: Image, magic: str = 'P6') -> bytes:
    height, width = shape(image)
    if magic == 'P6':
        header = f'P6\n{width} {height}\n255\n'.encode('ascii')
        raw = bytearray()
        for row in image:
            for px in row:
                raw.extend(px)
        return header + bytes(raw)
    elif magic == 'P3':
        lines = [f'P3\n{width} {height}\n255']
        for row in image:
            row_str = ' '.join(f'{p[0]} {p[1]} {p[2]}' for p in row)
            lines.append(row_str)
        return '\n'.join(lines).encode('ascii') + b'\n'
    else:
        raise ValueError(f'unsupported PPM magic: {magic}')

def load_image(source: str | Path | bytes) -> Image:
    if isinstance(source, (str, Path)):
        path = Path(source)
        raw = path.read_bytes()
    else:
        raw = bytes(source)
    if raw.startswith(b'P6') or raw.startswith(b'P3'):
        return load_ppm(raw)
    if raw.startswith(b'{'):
        data = json.loads(raw.decode('utf-8'))
        if 'image' in data:
            data = data['image']
        shape(data)
        return data
    try:
        from PIL import Image as PilImage  # noqa: PLC0415
        import io  # noqa: PLC0415
        pil_img = PilImage.open(io.BytesIO(raw)).convert('RGB')
        w, h = pil_img.size
        raw_rgb = pil_img.tobytes()
        out = []
        idx = 0
        for r in range(h):
            row = []
            for c in range(w):
                row.append((raw_rgb[idx], raw_rgb[idx + 1], raw_rgb[idx + 2]))
                idx += 3
            out.append(row)
        return out
    except ImportError:
        pass
    raise ValueError('unsupported or unhandled image format without Pillow (use .ppm or JSON)')

def save_image(image: Image, target: str | Path, format: str | None = None) -> None:
    height, width = shape(image)
    path = Path(target)
    fmt = format.lower() if format else path.suffix.lower().lstrip('.')
    if fmt in ('ppm', 'p6'):
        path.write_bytes(save_ppm(image, 'P6'))
        return
    if fmt == 'p3':
        path.write_bytes(save_ppm(image, 'P3'))
        return
    if fmt == 'json':
        path.write_text(json.dumps({'width': width, 'height': height, 'image': image}, indent=2), encoding='utf-8')
        return
    try:
        from PIL import Image as PilImage  # noqa: PLC0415
        pil_img = PilImage.new('RGB', (width, height))
        flat = [px for row in image for px in row]
        pil_img.putdata(flat)
        pil_img.save(str(path))
        return
    except ImportError:
        pass
    if fmt == 'png':
        # Fallback to PPM if PIL is not installed
        fallback = path.with_suffix('.ppm')
        fallback.write_bytes(save_ppm(image, 'P6'))
        return
    raise ValueError(f'cannot save format {fmt} without Pillow')
