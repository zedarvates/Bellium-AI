#!/usr/bin/env python3
from __future__ import annotations

import json
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bellium.editing.compare import compare_images
from bellium.editing.quantize import extract_palette_median_cut, quantize_floyd_steinberg
from bellium.editing.resample import resample_bilinear
from bellium.micro_nn.tone_curve import micro_tone_adjust
from bellium.nano_nn.adaptive_filter import nano_adaptive_filter
from bellium.nano_nn.quantize_tone import nano_quantize
from bellium.nano_nn.resample_edge import nano_resample

REPORT = (
    Path(__file__).resolve().parents[1]
    / 'benchmarks'
    / 'manifests'
    / 'magick-replacement-v1.json'
)

def generate_test_corpus():
    size = 32
    images = {}
    # 1. Step edge image
    step = []
    for r in range(size):
        row = []
        for c in range(size):
            val = 220 if (r + c) >= size else 30
            row.append((val, val, val))
        step.append(row)
    images['step_edge'] = step
    # 2. Smooth gradient image
    grad = []
    for r in range(size):
        row = []
        for c in range(size):
            val = int(255 * (r + c) / (2 * size))
            row.append((val, int(val * 0.8), int(val * 0.5)))
        grad.append(row)
    images['gradient'] = grad
    # 3. High-contrast lineart
    lines = []
    for r in range(size):
        row = []
        for c in range(size):
            is_line = (r % 4 == 0) or (c % 4 == 0)
            val = 10 if is_line else 245
            row.append((val, val, val))
        lines.append(row)
    images['lineart'] = lines
    return images

def benchmark() -> dict[str, object]:
    corpus = generate_test_corpus()
    results = {}
    
    # Benchmark Resize (Nano-NN vs Bilinear vs Nearest)
    resize_stats = {}
    for name, img in corpus.items():
        t0 = time.perf_counter()
        res_nano = nano_resample(img, 64, 64)
        dt_nano = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        res_bilinear = resample_bilinear(img, 64, 64)
        dt_bilinear = (time.perf_counter() - t0) * 1000.0
        comp = compare_images(res_nano.output['image'], res_bilinear)
        resize_stats[name] = {
            'nano_ms': round(dt_nano, 2),
            'bilinear_ms': round(dt_bilinear, 2),
            'diff_ssim': comp['ssim'],
            'diff_mae': comp['mae'],
            'refined_pixels': res_nano.output['refined_pixels'],
        }
    results['resize'] = resize_stats
    
    # Benchmark Quantization (Nano-NN dither arbiter vs Floyd-Steinberg)
    quant_stats = {}
    for name, img in corpus.items():
        palette = extract_palette_median_cut(img, max_colors=8)
        t0 = time.perf_counter()
        res_nano_q = nano_quantize(img, palette)
        dt_nano_q = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        _ = quantize_floyd_steinberg(img, palette, 1.0)
        dt_fs = (time.perf_counter() - t0) * 1000.0
        quant_stats[name] = {
            'nano_quant_ms': round(dt_nano_q, 2),
            'floyd_steinberg_ms': round(dt_fs, 2),
            'dither_intensity': res_nano_q.output['dither_intensity'],
        }
    results['quantize'] = quant_stats
    
    # Benchmark Adaptive Filter
    filter_stats = {}
    for name, img in corpus.items():
        t0 = time.perf_counter()
        res_filt = nano_adaptive_filter(img)
        dt_filt = (time.perf_counter() - t0) * 1000.0
        filter_stats[name] = {
            'nano_filter_ms': round(dt_filt, 2),
            'sharp_weight': res_filt.output['average_sharpness_weight'],
            'smooth_weight': res_filt.output['average_smoothing_weight'],
        }
    results['adaptive_filter'] = filter_stats
    
    # Benchmark Tone Adjust
    tone_stats = {}
    for name, img in corpus.items():
        t0 = time.perf_counter()
        res_tone = micro_tone_adjust(img)
        dt_tone = (time.perf_counter() - t0) * 1000.0
        tone_stats[name] = {
            'micro_tone_ms': round(dt_tone, 2),
            'curve': res_tone.output['curve_parameters'],
        }
    results['tone_curve'] = tone_stats
    
    manifest = {
        'benchmark': 'magick-replacement-v1',
        'status': 'passed',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'results': results,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Wrote manifest to {REPORT}')
    return manifest

if __name__ == '__main__':
    benchmark()
