from __future__ import annotations

from bellium.knn._image import Image, clamp_rgb, shape

def extract_channel(image: Image, channel: str = 'r') -> Image:
    height, width = shape(image)
    ch = channel.lower().strip()
    if ch not in ('r', 'g', 'b', 'luma', 'red', 'green', 'blue'):
        raise ValueError(f'unsupported channel {channel}; expected r, g, b or luma')
    out = []
    for row in image:
        out_row = []
        for px in row:
            if ch in ('r', 'red'):
                v = px[0]
            elif ch in ('g', 'green'):
                v = px[1]
            elif ch in ('b', 'blue'):
                v = px[2]
            else:  # luma
                v = clamp_rgb((2126 * px[0] + 7152 * px[1] + 722 * px[2]) / 10000.0)
            out_row.append((v, v, v))
        out.append(out_row)
    return out

def combine_channels(r_image: Image, g_image: Image, b_image: Image) -> Image:
    hr, wr = shape(r_image)
    hg, wg = shape(g_image)
    hb, wb = shape(b_image)
    if not (hr == hg == hb and wr == wg == wb):
        raise ValueError('channel images must share identical dimensions')
    out = []
    for r in range(hr):
        out_row = []
        for c in range(wr):
            vr = r_image[r][c][0]
            vg = g_image[r][c][1]
            vb = b_image[r][c][2]
            out_row.append((vr, vg, vb))
        out.append(out_row)
    return out
