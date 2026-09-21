import pytest
from copy import deepcopy

from bellium.editing.compare import compare_images, diff_image
from bellium.editing.channels import combine_channels, extract_channel
from bellium.editing.colorspace import delta_e_cie76, hsv_to_rgb, rgb_to_hsv
from bellium.editing.composite import composite
from bellium.editing.montage import create_montage
from bellium.editing.convolve import box_blur, emboss, gaussian_blur, laplacian, sharpen, sobel_edges
from bellium.editing.geometry import flip_horizontal, flip_vertical, pad_extent, rotate_180, rotate_270_cw, rotate_90_cw, trim_uniform
from bellium.editing.morphology import adaptive_local_threshold, close_mask, dilate_mask, erode_mask, open_mask, otsu_threshold
from bellium.editing.quantize import extract_palette_median_cut, quantize_floyd_steinberg, quantize_nearest
from bellium.editing.resample import resample_bilinear, resample_box, resample_nearest
from bellium.editing.tones import auto_gamma, contrast_stretch, histogram_equalize
from bellium.knn.adaptive_threshold import knn_threshold
from bellium.knn.filter_selector import select_filter_knn
from bellium.knn.palette_match import match_color_knn, quantize_palette_knn
from bellium.knn.resample_knn import knn_resample
from bellium.micro_nn.tone_curve import micro_tone_adjust
from bellium.nano_nn.adaptive_filter import nano_adaptive_filter
from bellium.nano_nn.quantize_tone import nano_quantize
from bellium.nano_nn.resample_edge import nano_resample
from bellium.specialists.magick import magick_process

def _sample_image(height=4, width=4, fill=(100, 150, 200)):
    return [[fill for _ in range(width)] for _ in range(height)]

def _pattern_image():
    return [
        [(10, 10, 10), (10, 10, 10), (200, 200, 200), (200, 200, 200)],
        [(10, 10, 10), (10, 10, 10), (200, 200, 200), (200, 200, 200)],
        [(50, 50, 50), (50, 50, 50), (150, 150, 150), (150, 150, 150)],
        [(50, 50, 50), (50, 50, 50), (150, 150, 150), (150, 150, 150)],
    ]

def test_resample_deterministic_and_specialists():
    img = _pattern_image()
    copy_img = deepcopy(img)
    nearest = resample_nearest(img, 8, 8)
    assert len(nearest) == 8 and len(nearest[0]) == 8
    bilinear = resample_bilinear(img, 8, 8)
    assert len(bilinear) == 8 and len(bilinear[0]) == 8
    box = resample_box(img, 2, 2)
    assert len(box) == 2 and len(box[0]) == 2
    assert img == copy_img
    knn_res = knn_resample(img, 8, 8)
    assert knn_res.output['image'] is not None
    assert not knn_res.abstained
    nano_res = nano_resample(img, 8, 8)
    assert nano_res.output['image'] is not None
    assert nano_res.confidence >= 0.9

def test_quantization_and_palette_matching():
    img = _pattern_image()
    palette = extract_palette_median_cut(img, max_colors=4)
    assert len(palette) <= 4
    q_near = quantize_nearest(img, palette)
    for row in q_near:
        for px in row:
            assert px in palette
    q_fs = quantize_floyd_steinberg(img, palette, dither_strength=0.8)
    for row in q_fs:
        for px in row:
            assert px in palette
    k_matches = match_color_knn((105, 145, 195), palette, k=2)
    assert len(k_matches) == min(2, len(palette))
    knn_q = quantize_palette_knn(img, palette)
    assert knn_q.output['active_colors'] > 0
    assert not knn_q.abstained
    nano_q = nano_quantize(img, palette)
    assert 0.0 <= nano_q.output['dither_intensity'] <= 1.0
    assert not nano_q.abstained

def test_convolutions_and_filtering():
    img = _pattern_image()
    g_blur = gaussian_blur(img, radius=1, sigma=1.0)
    assert len(g_blur) == 4
    b_blur = box_blur(img, radius=1)
    assert len(b_blur) == 4
    shp = sharpen(img, strength=0.8)
    assert len(shp) == 4
    sob = sobel_edges(img)
    assert len(sob) == 4
    emb = emboss(img)
    assert len(emb) == 4
    lap = laplacian(img)
    assert len(lap) == 4
    sel_filter = select_filter_knn(img)
    assert not sel_filter.abstained
    assert 'recommended_filter' in sel_filter.output
    nano_filt = nano_adaptive_filter(img)
    assert not nano_filt.abstained
    assert 0.0 <= nano_filt.output['average_sharpness_weight'] <= 1.0

def test_tones_and_curve_regression():
    img = _pattern_image()
    stretched = contrast_stretch(img)
    assert len(stretched) == 4
    equalized = histogram_equalize(img)
    assert len(equalized) == 4
    gamma_adj = auto_gamma(img, target_luma=128.0)
    assert len(gamma_adj) == 4
    micro_t = micro_tone_adjust(img)
    assert not micro_t.abstained
    assert 'curve_parameters' in micro_t.output

def test_morphology_and_thresholding():
    img = _pattern_image()
    mask, t = otsu_threshold(img)
    assert len(mask) == 4
    assert 0 <= t <= 255
    local_mask = adaptive_local_threshold(img, window_size=3)
    assert len(local_mask) == 4
    dilated = dilate_mask(mask, radius=1)
    assert len(dilated) == 4
    eroded = erode_mask(mask, radius=1)
    assert len(eroded) == 4
    opened = open_mask(mask, radius=1)
    assert len(opened) == 4
    closed = close_mask(mask, radius=1)
    assert len(closed) == 4
    knn_t = knn_threshold(img)
    assert not knn_t.abstained
    assert 'strategy' in knn_t.output
    flat_img = _sample_image(4, 4, fill=(128, 128, 128))
    flat_res = knn_threshold(flat_img)
    assert flat_res.abstained

def test_comparison_and_metrics():
    img1 = _sample_image(4, 4, fill=(100, 100, 100))
    img2 = _sample_image(4, 4, fill=(100, 100, 100))
    metrics_same = compare_images(img1, img2)
    assert metrics_same['mae'] == 0.0
    assert metrics_same['ssim'] == 1.0
    assert metrics_same['mismatch_pixels'] == 0.0
    img3 = _sample_image(4, 4, fill=(120, 120, 120))
    metrics_diff = compare_images(img1, img3)
    assert metrics_diff['mae'] == 20.0
    assert metrics_diff['mismatch_pixels'] == 16.0
    diff = diff_image(img1, img3)
    assert diff[0][0] == (255, 0, 0)

def test_geometry_transforms():
    img = _pattern_image()
    assert flip_horizontal(flip_horizontal(img)) == img
    assert flip_vertical(flip_vertical(img)) == img
    rot = rotate_90_cw(rotate_90_cw(rotate_90_cw(rotate_90_cw(img))))
    assert rot == img
    assert rotate_180(img) == rotate_90_cw(rotate_90_cw(img))
    assert rotate_270_cw(img) == rotate_90_cw(rotate_90_cw(rotate_90_cw(img)))
    padded = pad_extent(img, 6, 6, background=(0, 0, 0))
    assert len(padded) == 6 and len(padded[0]) == 6
    trimmed, bbox = trim_uniform(padded)
    assert bbox == (1, 1, 5, 5)
    assert trimmed == img

def test_magick_process_unified_specialist():
    img = _pattern_image()
    res_resize = magick_process({'image': img, 'operation': 'resize', 'height': 8, 'width': 8, 'method': 'nano_edge'})
    assert not res_resize.abstained
    res_quant = magick_process({'image': img, 'operation': 'quantize', 'colors': 4, 'dither': 'nano_tone'})
    assert not res_quant.abstained
    res_filt = magick_process({'image': img, 'operation': 'filter', 'filter_type': 'nano_adaptive'})
    assert not res_filt.abstained
    res_tone = magick_process({'image': img, 'operation': 'tone', 'mode': 'micro_nn'})
    assert not res_tone.abstained
    res_thresh = magick_process({'image': img, 'operation': 'threshold', 'mode': 'knn'})
    assert not res_thresh.abstained
    res_morph = magick_process({'image': img, 'operation': 'morphology', 'morph_op': 'dilate', 'radius': 1})
    assert not res_morph.abstained
    res_comp = magick_process({'image': img, 'operation': 'compare', 'image_b': img})
    assert res_comp.output['metrics']['mae'] == 0.0
    res_trans = magick_process({'image': img, 'operation': 'transform', 'transform_type': 'rotate180'})
    assert not res_trans.abstained
    res_comp_layer = magick_process({'image': img, 'operation': 'composite', 'overlay': img, 'mode': 'multiply'})
    assert not res_comp_layer.abstained
    res_chan_ex = magick_process({'image': img, 'operation': 'channel', 'action': 'extract', 'channel': 'luma'})
    assert not res_chan_ex.abstained
    res_chan_comb = magick_process({'image': img, 'operation': 'channel', 'action': 'combine', 'g_image': img, 'b_image': img})
    assert not res_chan_comb.abstained
    with pytest.raises(ValueError):
        magick_process({'image': img, 'operation': 'unknown_op'})

def test_compositing_and_channels():
    base = [[(100, 100, 100), (200, 200, 200)], [(50, 50, 50), (150, 150, 150)]]
    over = [[(50, 50, 50), (100, 100, 100)], [(25, 25, 25), (75, 75, 75)]]
    mult = composite(base, over, mode='multiply')
    assert mult[0][0] == (20, 20, 20)  # 100/255 * 50/255 * 255 = 19.608 -> rounds to 20
    screen = composite(base, over, mode='screen')
    assert screen[0][0][0] > base[0][0][0]
    r_ch = extract_channel(base, 'r')
    assert r_ch[0][0] == (100, 100, 100)
    comb = combine_channels(base, base, base)
    assert comb == base

def test_colorspace_and_montage():
    red_rgb = (255, 0, 0)
    h, s, v = rgb_to_hsv(red_rgb)
    assert h == 0.0 and s == 1.0 and v == 1.0
    assert hsv_to_rgb(h, s, v) == red_rgb
    c1 = (200, 100, 50)
    c2 = (200, 100, 50)
    assert delta_e_cie76(c1, c2) == 0.0
    c3 = (205, 100, 50)
    d_e = delta_e_cie76(c1, c3)
    assert 0.0 < d_e < 10.0
    img_a = [[(255, 0, 0)]]
    img_b = [[(0, 255, 0)]]
    img_c = [[(0, 0, 255)]]
    img_d = [[(255, 255, 0)]]
    mont = create_montage([img_a, img_b, img_c, img_d], columns=2, rows=2, padding=1)
    assert len(mont) == 5 and len(mont[0]) == 5
    res_mont = magick_process({'image': img_a, 'images': [img_a, img_b], 'operation': 'montage', 'columns': 2, 'rows': 1})
    assert not res_mont.abstained
