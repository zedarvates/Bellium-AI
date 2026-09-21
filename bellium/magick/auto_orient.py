from __future__ import annotations

from bellium.editing.geometry import flip_horizontal, flip_vertical, rotate_180, rotate_270_cw, rotate_90_cw
from bellium.knn._image import Image, shape

def auto_orient(image: Image, orientation: int = 1) -> Image:
    shape(image)
    if orientation == 1:
        return [list(r) for r in image]
    elif orientation == 2:
        return flip_horizontal(image)
    elif orientation == 3:
        return rotate_180(image)
    elif orientation == 4:
        return flip_vertical(image)
    elif orientation == 5:
        return rotate_270_cw(flip_horizontal(image))
    elif orientation == 6:
        return rotate_90_cw(image)
    elif orientation == 7:
        return rotate_90_cw(flip_horizontal(image))
    elif orientation == 8:
        return rotate_270_cw(image)
    else:
        return [list(r) for r in image]
