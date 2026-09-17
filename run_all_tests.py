#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import subprocess
import time

# Configure stdout for safe encoding on all platforms
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def run_all_tests():
    root = os.path.dirname(os.path.abspath(__file__))
    env = os.environ.copy()
    env['PYTHONPATH'] = root
    
    test_suites = [
        ('Legacy Micro-NN Features', 'legacy/micro_nn/test_features.py'),
        ('Legacy Micro-NN Inferences & Routing', 'legacy/micro_nn/test_botte_nn.py'),
        ('Legacy Micro-NN Calibration', 'legacy/micro_nn/test_calibration.py'),
        ('Legacy k-NN Asset Quality Memory', 'legacy/knn_asset_quality/test_asset_quality.py'),
        ('Bellium Native: Cutout & Background Normalizer', 'bellium/cutout/test_cutout.py'),
        ('Bellium Native: Patch k-NN Inpaint & Escalation', 'bellium/inpaint/test_inpaint.py'),
        ('Bellium Native: Phoneme k-NN & Linguistic Reconstruction', 'bellium/language/test_phoneme.py'),
        ('Bellium Native: Tiered Specialist Router & Abstention', 'bellium/routing/test_routing.py'),
        ('Bellium Native: End-to-End Asset Prep Pipeline', 'bellium/pipeline/test_pipeline.py'),
        ('Bellium Native: Visual Anomaly & Emergency Gates', 'bellium/vision/test_anomaly.py'),
        ('Bellium Native: Voice Activity & Audio Quality Gates', 'bellium/audio/test_audio.py'),
    ]
    
    print('=' * 65)
    print('   Bellium AI Test Suite - Verification Gate')
    print('=' * 65)
    
    passed_count = 0
    failed_count = 0
    t0 = time.perf_counter()
    
    for name, rel_path in test_suites:
        abs_path = os.path.join(root, rel_path)
        if not os.path.exists(abs_path):
            print(f'[FAIL] MISSING SUITE: {name} ({rel_path})')
            failed_count += 1
            continue
            
        t_start = time.perf_counter()
        proc = subprocess.run([sys.executable, abs_path], cwd=root, env=env, capture_output=True, text=True)
        elapsed = (time.perf_counter() - t_start) * 1000
        
        if proc.returncode == 0:
            print(f'  [PASS] {name:<50} ({elapsed:6.1f} ms)')
            passed_count += 1
        else:
            print(f'  [FAIL] {name:<50} (code {proc.returncode})')
            print('STDERR:', proc.stderr)
            print('STDOUT:', proc.stdout)
            failed_count += 1
            
    total_elapsed = (time.perf_counter() - t0) * 1000
    print('-' * 65)
    print(f'Total: {len(test_suites)} suites | {passed_count} passed | {failed_count} failed ({total_elapsed:.1f} ms)')
    print('=' * 65)
    
    if failed_count > 0:
        sys.exit(1)
    else:
        print('All Bellium AI validation gates passed successfully.')
        sys.exit(0)

if __name__ == '__main__':
    run_all_tests()
