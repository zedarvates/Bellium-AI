from __future__ import annotations

from typing import Any

from bellium.contracts.schema import AuthorityMode, SpecialistResult
from bellium.editing.channels import combine_channels, extract_channel
from bellium.editing.composite import composite
from bellium.editing.montage import create_montage
from bellium.editing.compare import compare_images, diff_image
from bellium.editing.convolve import emboss, gaussian_blur, sharpen, sobel_edges
from bellium.editing.geometry import flip_horizontal, flip_vertical, pad_extent, rotate_180, rotate_270_cw, rotate_90_cw, trim_uniform
from bellium.editing.morphology import close_mask, dilate_mask, erode_mask, open_mask, otsu_threshold
from bellium.editing.quantize import extract_palette_median_cut, quantize_floyd_steinberg, quantize_nearest
from bellium.editing.resample import resample_bilinear, resample_box, resample_nearest
from bellium.editing.tones import auto_gamma, contrast_stretch, histogram_equalize
from bellium.knn._image import shape
from bellium.knn.adaptive_threshold import knn_threshold
from bellium.knn.filter_selector import select_filter_knn
from bellium.specialists.vectorize import raster_to_svg
from bellium.knn.palette_match import quantize_palette_knn
from bellium.knn.resample_knn import knn_resample
from bellium.micro_nn.tone_curve import micro_tone_adjust
from bellium.nano_nn.adaptive_filter import nano_adaptive_filter
from bellium.nano_nn.quantize_tone import nano_quantize
from bellium.nano_nn.resample_edge import nano_resample

SPECIALIST_ID = 'bellium/hybrid/magick-replacement:v0'

def magick_process(query: dict[str, Any]) -> SpecialistResult:
    image = query.get('image')
    if image is None:
        raise ValueError('query needs image')
    shape(image)
    op = query.get('operation')
    if not op:
        raise ValueError('query needs operation')
    
    if op == 'resize':
        target_h = query.get('height')
        target_w = query.get('width')
        method = query.get('method', 'nano_edge')
        if method == 'nano_edge':
            return nano_resample(image, target_h, target_w)
        elif method == 'knn':
            return knn_resample(image, target_h, target_w)
        elif method == 'bilinear':
            res = resample_bilinear(image, target_h, target_w)
            return SpecialistResult(SPECIALIST_ID, {'image': res, 'method': 'bilinear'}, 0.8, False, AuthorityMode.CONSULTATIVE)
        elif method == 'nearest':
            res = resample_nearest(image, target_h, target_w)
            return SpecialistResult(SPECIALIST_ID, {'image': res, 'method': 'nearest'}, 0.7, False, AuthorityMode.CONSULTATIVE)
        elif method == 'box':
            res = resample_box(image, target_h, target_w)
            return SpecialistResult(SPECIALIST_ID, {'image': res, 'method': 'box'}, 0.8, False, AuthorityMode.CONSULTATIVE)
        else:
            raise ValueError(f'unknown resize method: {method}')
            
    elif op == 'quantize':
        colors = query.get('colors', 16)
        palette = query.get('palette')
        if palette is None:
            palette = extract_palette_median_cut(image, max_colors=colors)
        dither = query.get('dither', 'nano_tone')
        if dither == 'nano_tone':
            return nano_quantize(image, palette)
        elif dither == 'knn':
            return quantize_palette_knn(image, palette)
        elif dither == 'floyd_steinberg':
            res = quantize_floyd_steinberg(image, palette, dither_strength=1.0)
            return SpecialistResult(SPECIALIST_ID, {'image': res, 'palette': palette, 'dither': 'floyd_steinberg'}, 0.85, False, AuthorityMode.CONSULTATIVE)
        elif dither == 'none':
            res = quantize_nearest(image, palette)
            return SpecialistResult(SPECIALIST_ID, {'image': res, 'palette': palette, 'dither': 'none'}, 0.8, False, AuthorityMode.CONSULTATIVE)
        else:
            raise ValueError(f'unknown dither mode: {dither}')
            
    elif op == 'filter':
        filter_type = query.get('filter_type', 'auto')
        if filter_type == 'auto':
            return select_filter_knn(image)
        elif filter_type == 'nano_adaptive':
            return nano_adaptive_filter(image)
        elif filter_type == 'sharpen':
            res = sharpen(image, strength=query.get('strength', 1.0))
            return SpecialistResult(SPECIALIST_ID, {'image': res, 'filter': 'sharpen'}, 0.85, False, AuthorityMode.CONSULTATIVE)
        elif filter_type == 'blur':
            res = gaussian_blur(image, radius=query.get('radius', 1), sigma=query.get('sigma', 1.0))
            return SpecialistResult(SPECIALIST_ID, {'image': res, 'filter': 'gaussian_blur'}, 0.85, False, AuthorityMode.CONSULTATIVE)
        elif filter_type == 'edge':
            res = sobel_edges(image)
            return SpecialistResult(SPECIALIST_ID, {'image': res, 'filter': 'sobel'}, 0.85, False, AuthorityMode.CONSULTATIVE)
        elif filter_type == 'emboss':
            res = emboss(image)
            return SpecialistResult(SPECIALIST_ID, {'image': res, 'filter': 'emboss'}, 0.85, False, AuthorityMode.CONSULTATIVE)
        else:
            raise ValueError(f'unknown filter type: {filter_type}')
            
    elif op == 'tone':
        mode = query.get('mode', 'micro_nn')
        if mode == 'micro_nn':
            return micro_tone_adjust(image)
        elif mode == 'stretch':
            res = contrast_stretch(image)
            return SpecialistResult(SPECIALIST_ID, {'image': res, 'mode': 'contrast_stretch'}, 0.85, False, AuthorityMode.CONSULTATIVE)
        elif mode == 'equalize':
            res = histogram_equalize(image)
            return SpecialistResult(SPECIALIST_ID, {'image': res, 'mode': 'histogram_equalize'}, 0.8, False, AuthorityMode.CONSULTATIVE)
        elif mode == 'gamma':
            res = auto_gamma(image, target_luma=query.get('target_luma', 128.0))
            return SpecialistResult(SPECIALIST_ID, {'image': res, 'mode': 'auto_gamma'}, 0.8, False, AuthorityMode.CONSULTATIVE)
        else:
            raise ValueError(f'unknown tone mode: {mode}')
            
    elif op == 'threshold':
        mode = query.get('mode', 'knn')
        if mode == 'knn':
            return knn_threshold(image)
        elif mode == 'otsu':
            mask, t = otsu_threshold(image)
            return SpecialistResult(SPECIALIST_ID, {'mask': mask, 'threshold': t, 'mode': 'otsu'}, 0.8, False, AuthorityMode.CONSULTATIVE)
        else:
            raise ValueError(f'unknown threshold mode: {mode}')
            
    elif op == 'morphology':
        mask = query.get('mask')
        if mask is None:
            mask, _ = otsu_threshold(image)
        morph_op = query.get('morph_op', 'dilate')
        radius = query.get('radius', 1)
        if morph_op == 'dilate':
            res_mask = dilate_mask(mask, radius)
        elif morph_op == 'erode':
            res_mask = erode_mask(mask, radius)
        elif morph_op == 'open':
            res_mask = open_mask(mask, radius)
        elif morph_op == 'close':
            res_mask = close_mask(mask, radius)
        else:
            raise ValueError(f'unknown morphology op: {morph_op}')
        return SpecialistResult(SPECIALIST_ID, {'mask': res_mask, 'morph_op': morph_op, 'radius': radius}, 0.9, False, AuthorityMode.CONSULTATIVE)
        
    elif op == 'compare':
        image_b = query.get('image_b')
        if image_b is None:
            raise ValueError('compare needs image_b')
        metrics = compare_images(image, image_b)
        diff = diff_image(image, image_b)
        return SpecialistResult(SPECIALIST_ID, {'metrics': metrics, 'diff_image': diff}, 0.95, False, AuthorityMode.CONSULTATIVE)
        
    elif op == 'transform':
        trans = query.get('transform_type')
        if trans == 'flip_h':
            res = flip_horizontal(image)
        elif trans == 'flip_v':
            res = flip_vertical(image)
        elif trans == 'rotate90':
            res = rotate_90_cw(image)
        elif trans == 'rotate180':
            res = rotate_180(image)
        elif trans == 'rotate270':
            res = rotate_270_cw(image)
        elif trans == 'pad':
            res = pad_extent(image, query.get('target_height'), query.get('target_width'), background=query.get('background', (0,0,0)))
        elif trans == 'trim':
            res, box = trim_uniform(image, tolerance=query.get('tolerance', 0.0))
            return SpecialistResult(SPECIALIST_ID, {'image': res, 'bbox': box, 'transform': 'trim'}, 0.9, False, AuthorityMode.CONSULTATIVE)
        else:
            raise ValueError(f'unknown transform: {trans}')
        return SpecialistResult(SPECIALIST_ID, {'image': res, 'transform': trans}, 0.95, False, AuthorityMode.CONSULTATIVE)
        
    elif op == 'composite':
        overlay = query.get('overlay')
        if overlay is None:
            raise ValueError('composite operation needs overlay image')
        mode = query.get('mode', 'over')
        opacity = query.get('opacity', 1.0)
        mask = query.get('mask')
        offset = query.get('offset', (0, 0))
        res = composite(image, overlay, mode=mode, opacity=opacity, mask=mask, offset=offset)
        return SpecialistResult(
            SPECIALIST_ID,
            {'image': res, 'mode': mode, 'opacity': opacity, 'offset': offset},
            0.95,
            False,
            AuthorityMode.CONSULTATIVE,
        )
        
    elif op == 'channel':
        action = query.get('action', 'extract')
        if action == 'extract':
            ch = query.get('channel', 'luma')
            res = extract_channel(image, channel=ch)
            return SpecialistResult(SPECIALIST_ID, {'image': res, 'channel': ch, 'action': 'extract'}, 0.95, False, AuthorityMode.CONSULTATIVE)
        elif action == 'combine':
            g_img = query.get('g_image')
            b_img = query.get('b_image')
            if g_img is None or b_img is None:
                raise ValueError('combine action needs image (as red), g_image and b_image')
            res = combine_channels(image, g_img, b_img)
            return SpecialistResult(SPECIALIST_ID, {'image': res, 'action': 'combine'}, 0.95, False, AuthorityMode.CONSULTATIVE)
        else:
            raise ValueError(f'unknown channel action: {action}')
        
    elif op == 'montage':
        images = query.get('images', [image])
        columns = query.get('columns')
        rows = query.get('rows')
        tile_size = query.get('tile_size')
        padding = query.get('padding', 2)
        bg = query.get('background', (20, 20, 20))
        res = create_montage(images, columns=columns, rows=rows, tile_size=tile_size, padding=padding, background=bg)
        return SpecialistResult(SPECIALIST_ID, {'image': res, 'tiles_count': len(images)}, 0.95, False, AuthorityMode.CONSULTATIVE)
        
    elif op == 'vectorize':
        return raster_to_svg(query)
        
    else:
        raise ValueError(f'unknown magick operation: {op}')
