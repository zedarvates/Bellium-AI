from __future__ import annotations


Rgb = tuple[int, int, int]
Image = list[list[Rgb]]
Mask = list[list[int]]


def shape(image: Image) -> tuple[int, int]:
    if not image or not image[0]:
        raise ValueError("image is empty")
    height = len(image)
    width = len(image[0])
    if any(len(row) != width for row in image):
        raise ValueError("image rows must have equal width")
    for row in image:
        for pixel in row:
            if not isinstance(pixel, (tuple, list)) or len(pixel) != 3 or any(
                isinstance(channel, bool) or not isinstance(channel, int) or not 0 <= channel <= 255
                for channel in pixel
            ):
                raise ValueError("image pixels must contain three integer RGB channels in [0, 255]")
    return height, width


def clamp_rgb(value: float) -> int:
    return max(0, min(255, int(round(value))))


def copy_image(image: Image) -> Image:
    return [list(row) for row in image]


def validate_mask(image: Image, mask: Mask) -> None:
    height, width = shape(image)
    if len(mask) != height or any(len(row) != width for row in mask):
        raise ValueError("mask shape must match image")
    for row in mask:
        for value in row:
            if value not in (0, 1):
                raise ValueError("mask values must be 0 or 1")
