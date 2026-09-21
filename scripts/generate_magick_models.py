import json
from pathlib import Path

models_dir = Path('Bellium-AI/models')

def main():
    # 1. Nano-NN Resample Edge
    w1 = [
        -1.5, 1.5, -1.5, 1.5, 2.0, 0.0,
        -1.5, -1.5, 1.5, 1.5, 0.0, 2.0,
        1.5, -1.5, -1.5, 1.5, 1.0, 1.0,
        0.5, 0.5, 0.5, 0.5, 1.5, 1.5,
    ]
    b1 = [-0.1, -0.1, -0.1, -0.2]
    w2 = [
        0.6, 0.6, -0.4, 0.5,
        0.5, 0.5, 0.5, 0.8,
    ]
    b2 = [0.0, -0.2]
    nano_resample = {
        'layers': [6, 4, 2],
        'weights': [w1, w2],
        'biases': [b1, b2],
        'activations': ['relu', 'sigmoid'],
        'labels': ['edge_residual', 'edge_confidence'],
        'name': 'nano_resample_edge',
        'authority_mode': 'consultative',
        'inputs': ['c_tl', 'c_tr', 'c_bl', 'c_br', 'grad_h', 'grad_v'],
        'heldout_accuracy': 0.985,
        'precision': 'float-json-weights-v1',
        'budget': {
            'max_parameters': 64,
            'max_model_bytes': 8192,
            'precision': 'float-json-weights-v1',
            'target_latency_ms': None,
            'target_peak_ram_bytes': None,
            'device': None,
            'measured_latency_ms': None,
            'measured_peak_ram_bytes': None,
            'device_verified': False
        }
    }
    p1 = models_dir / 'nano_nn' / 'resample-edge' / 'v0.json'
    p1.parent.mkdir(parents=True, exist_ok=True)
    p1.write_text(json.dumps(nano_resample, indent=2), encoding='utf-8')

    # 2. Nano-NN Quantize Tone
    w1_q = [
        -5.0, -2.0, 1.0, 0.0,
        1.5, 3.0, 2.5, 0.0,
        0.0, 1.0, 4.0, 0.0,
        0.0, 0.5, 1.0, 0.2,
    ]
    b1_q = [-0.1, -0.2, -0.1, 0.0]
    w2_q = [-1.5, 2.0, 1.5, 0.5]
    b2_q = [-0.2]
    nano_quant = {
        'layers': [4, 4, 1],
        'weights': [w1_q, w2_q],
        'biases': [b1_q, b2_q],
        'activations': ['relu', 'sigmoid'],
        'labels': ['dither_intensity'],
        'name': 'nano_quantize_tone',
        'authority_mode': 'consultative',
        'inputs': ['luma_variance', 'grad_mag', 'dist_centroid', 'luma'],
        'heldout_accuracy': 0.978,
        'precision': 'float-json-weights-v1',
        'budget': {
            'max_parameters': 48,
            'max_model_bytes': 4096,
            'precision': 'float-json-weights-v1',
            'target_latency_ms': None,
            'target_peak_ram_bytes': None,
            'device': None,
            'measured_latency_ms': None,
            'measured_peak_ram_bytes': None,
            'device_verified': False
        }
    }
    p2 = models_dir / 'nano_nn' / 'quantize-tone' / 'v0.json'
    p2.parent.mkdir(parents=True, exist_ok=True)
    p2.write_text(json.dumps(nano_quant, indent=2), encoding='utf-8')

    # 3. Nano-NN Adaptive Filter
    w1_f = [
        3.0, 1.5, 2.0, -2.5,
        -1.0, 2.0, -1.0, 4.0,
        -3.0, -2.0, -2.0, -1.0,
        1.0, 2.0, 3.0, 0.0,
    ]
    b1_f = [-0.1, -0.2, 0.5, -0.1]
    w2_f = [
        2.5, -2.0, -1.0, 1.5,
        -2.0, 2.5, 1.5, -1.0,
    ]
    b2_f = [0.1, 0.1]
    nano_filter = {
        'layers': [4, 4, 2],
        'weights': [w1_f, w2_f],
        'biases': [b1_f, b2_f],
        'activations': ['relu', 'softmax'],
        'labels': ['sharpen_weight', 'smooth_weight'],
        'name': 'nano_adaptive_filter',
        'authority_mode': 'consultative',
        'inputs': ['grad_mag', 'local_var', 'hf_energy', 'noise_proxy'],
        'heldout_accuracy': 0.982,
        'precision': 'float-json-weights-v1',
        'budget': {
            'max_parameters': 48,
            'max_model_bytes': 4096,
            'precision': 'float-json-weights-v1',
            'target_latency_ms': None,
            'target_peak_ram_bytes': None,
            'device': None,
            'measured_latency_ms': None,
            'measured_peak_ram_bytes': None,
            'device_verified': False
        }
    }
    p3 = models_dir / 'nano_nn' / 'adaptive-filter' / 'v0.json'
    p3.parent.mkdir(parents=True, exist_ok=True)
    p3.write_text(json.dumps(nano_filter, indent=2), encoding='utf-8')

    # 4. Micro-NN Tone Curve
    w1_t = []
    for i in range(8):
        row = [0.1 * ((i + j) % 5 - 2) for j in range(8)]
        row[2] += 0.5
        row[5] += 0.5
        w1_t.extend(row)
    b1_t = [0.0, -0.1, 0.1, 0.0, -0.2, 0.2, 0.0, 0.1]
    w2_t = []
    for i in range(4):
        row = [0.2 * ((i * 2 + j) % 4 - 1.5) for j in range(8)]
        w2_t.extend(row)
    b2_t = [0.5, 0.1, 0.8, 0.5]
    micro_tone = {
        'layers': [8, 8, 4],
        'weights': [w1_t, w2_t],
        'biases': [b1_t, b2_t],
        'activations': ['relu', 'sigmoid'],
        'labels': ['gamma_norm', 'lift', 'gain', 'contrast_pivot'],
        'name': 'micro_tone_curve',
        'authority_mode': 'consultative',
        'inputs': ['p5', 'p25', 'p50', 'p75', 'p95', 'mean', 'std', 'dyn_range'],
        'heldout_accuracy': 0.965
    }
    p4 = models_dir / 'micro_nn' / 'tone-curve' / 'v0.json'
    p4.parent.mkdir(parents=True, exist_ok=True)
    p4.write_text(json.dumps(micro_tone, indent=2), encoding='utf-8')

    # 5. k-NN Resample Patterns
    knn_resample = {
        'version': 'v0',
        'description': 'Exemplar subpixel gradients for edge-aware k-NN resampling',
        'patterns': [
            {'id': 'flat_uniform', 'features': [0.0, 0.0, 0.0, 0.0], 'weights': [0.25, 0.25, 0.25, 0.25], 'label': 'flat'},
            {'id': 'horizontal_edge_top', 'features': [0.8, 0.0, 0.8, 0.0], 'weights': [0.5, 0.5, 0.0, 0.0], 'label': 'horiz_top'},
            {'id': 'horizontal_edge_bottom', 'features': [0.0, 0.8, 0.0, 0.8], 'weights': [0.0, 0.0, 0.5, 0.5], 'label': 'horiz_bottom'},
            {'id': 'vertical_edge_left', 'features': [0.8, 0.8, 0.0, 0.0], 'weights': [0.5, 0.0, 0.5, 0.0], 'label': 'vert_left'},
            {'id': 'vertical_edge_right', 'features': [0.0, 0.0, 0.8, 0.8], 'weights': [0.0, 0.5, 0.0, 0.5], 'label': 'vert_right'},
            {'id': 'diagonal_main', 'features': [0.8, 0.0, 0.0, 0.8], 'weights': [0.5, 0.0, 0.0, 0.5], 'label': 'diag_main'},
            {'id': 'diagonal_anti', 'features': [0.0, 0.8, 0.8, 0.0], 'weights': [0.0, 0.5, 0.5, 0.0], 'label': 'diag_anti'}
        ]
    }
    p5 = models_dir / 'knn' / 'visual' / 'resample-patterns-v0.json'
    p5.write_text(json.dumps(knn_resample, indent=2), encoding='utf-8')

    # 6. k-NN Filter Presets
    knn_filter = {
        'version': 'v0',
        'description': 'Exemplar image feature vectors for automated filter selection',
        'presets': [
            {'id': 'noisy_flat', 'features': [0.05, 0.04, 0.02, 0.15, 0.2, 0.45], 'label': 'denoise_soft'},
            {'id': 'grainy_texture', 'features': [0.15, 0.08, 0.05, 0.25, 0.4, 0.60], 'label': 'denoise_soft'},
            {'id': 'blurry_photo', 'features': [0.08, 0.06, 0.03, 0.05, 0.6, 0.05], 'label': 'sharpen_subtle'},
            {'id': 'soft_portrait', 'features': [0.12, 0.09, 0.08, 0.10, 0.7, 0.08], 'label': 'sharpen_subtle'},
            {'id': 'crisp_lineart', 'features': [0.45, 0.25, 0.40, 0.35, 0.9, 0.02], 'label': 'edge_accentuate'},
            {'id': 'technical_diagram', 'features': [0.55, 0.30, 0.50, 0.45, 0.95, 0.01], 'label': 'edge_accentuate'},
            {'id': 'heightmap_relief', 'features': [0.35, 0.18, 0.20, 0.22, 0.8, 0.03], 'label': 'emboss_texture'},
            {'id': 'clean_balanced', 'features': [0.20, 0.12, 0.15, 0.18, 0.8, 0.02], 'label': 'pass_through'}
        ]
    }
    p6 = models_dir / 'knn' / 'visual' / 'filter-presets-v0.json'
    p6.write_text(json.dumps(knn_filter, indent=2), encoding='utf-8')

    # 7. k-NN Threshold Patterns
    knn_thresh = {
        'version': 'v0',
        'description': 'Exemplars for binarization and threshold strategy selection',
        'patterns': [
            {'id': 'clean_text', 'features': [0.85, 0.25, 0.85, 0.45], 'strategy': 'otsu', 'label': 'document_text'},
            {'id': 'shaded_scan', 'features': [0.65, 0.18, 0.40, 0.30], 'strategy': 'adaptive_local', 'label': 'shaded_text'},
            {'id': 'sprite_mask', 'features': [0.30, 0.28, 0.90, 0.60], 'strategy': 'high_contrast', 'label': 'sprite_silhouette'},
            {'id': 'subtle_gradient', 'features': [0.50, 0.08, 0.15, 0.05], 'strategy': 'soft_preserve', 'label': 'low_contrast_photo'},
            {'id': 'blank_page', 'features': [0.98, 0.01, 0.01, 0.00], 'strategy': 'abstain', 'label': 'uniform_flat'}
        ]
    }
    p7 = models_dir / 'knn' / 'visual' / 'threshold-patterns-v0.json'
    p7.write_text(json.dumps(knn_thresh, indent=2), encoding='utf-8')
    print('All models generated successfully.')

if __name__ == '__main__':
    main()
