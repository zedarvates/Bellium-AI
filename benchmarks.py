#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import time
import math
from dataclasses import dataclass
from PIL import Image, ImageDraw

# Ensure safe console printing
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bellium.cutout import extract_foreground
from bellium.inpaint import inpaint_patch_knn
from bellium.language import PhonemeMemory, EvidenceLevel
from bellium.routing import HybridRouter, ToolCapability, TaskRequirement, EscalationTier
from bellium.vision import detect_visual_anomalies
from bellium.audio import detect_voice_activity
from bellium.adapters import repair_and_validate_json, SchemaRule


@dataclass
class BenchmarkMetric:
    module: str
    operation: str
    iterations: int
    mean_ms: float
    min_ms: float
    max_ms: float
    p95_ms: float
    ops_per_sec: float
    budget_ms: float
    verdict: str  # 'PASS_WITHIN_BUDGET', 'OVER_BUDGET'


def time_function(fn, iterations=20, warmup=3):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    mean_t = sum(times) / len(times)
    min_t = times[0]
    max_t = times[-1]
    p95_idx = int(math.ceil(0.95 * len(times))) - 1
    p95_t = times[p95_idx]
    ops = 1000.0 / mean_t if mean_t > 0 else 0.0
    return mean_t, min_t, max_t, p95_t, ops


def run_benchmarks():
    print('=' * 75)
    print('   Bellium AI - Micro-Specialist Latency & Throughput Benchmark')
    print('=' * 75)
    
    results = []
    
    # 1. Cutout benchmark (128x128 image with shape)
    im128 = Image.new('RGB', (128, 128), (255, 255, 255))
    draw1 = ImageDraw.Draw(im128)
    draw1.rectangle([30, 30, 98, 98], fill=(0, 120, 255))
    mean_t, min_t, max_t, p95_t, ops = time_function(lambda: extract_foreground(im128, tolerance=25), iterations=25)
    v = 'PASS_WITHIN_BUDGET' if p95_t <= 30.0 else 'OVER_BUDGET'
    results.append(BenchmarkMetric('bellium.cutout', 'extract_foreground (128x128)', 25, mean_t, min_t, max_t, p95_t, ops, 30.0, v))
    
    # 2. Patch k-NN Inpainting benchmark (64x64 with 10x10 hole)
    im64 = Image.new('RGB', (64, 64), (10, 150, 80))
    mask64 = Image.new('L', (64, 64), 0)
    draw_m = ImageDraw.Draw(mask64)
    draw_m.rectangle([25, 25, 35, 35], fill=255)
    mean_t, min_t, max_t, p95_t, ops = time_function(lambda: inpaint_patch_knn(im64, mask64, patch_size=5, search_radius=15), iterations=25)
    v = 'PASS_WITHIN_BUDGET' if p95_t <= 20.0 else 'OVER_BUDGET'
    results.append(BenchmarkMetric('bellium.inpaint', 'inpaint_patch_knn (64x64, 10x10 hole)', 25, mean_t, min_t, max_t, p95_t, ops, 20.0, v))
    
    # 3. Phoneme Memory query benchmark
    mem = PhonemeMemory()
    for i in range(50):
        mem.register_phoneme(f'p_{i}', 'fra', [0.1 * (i % 10)] * 7, EvidenceLevel.ATTESTED, 'corpus')
    q_vec = [0.45] * 7
    mean_t, min_t, max_t, p95_t, ops = time_function(lambda: mem.find_nearest_neighbors(q_vec, k=3), iterations=100)
    v = 'PASS_WITHIN_BUDGET' if p95_t <= 2.0 else 'OVER_BUDGET'
    results.append(BenchmarkMetric('bellium.language', 'k-NN retrieval (50 phonemes, k=3)', 100, mean_t, min_t, max_t, p95_t, ops, 2.0, v))
    
    # 4. Hybrid Router decision benchmark
    router = HybridRouter()
    router.register_tool(ToolCapability('bellium/cutout', EscalationTier.DETERMINISTIC, {'image'}, {'cutout'}))
    router.register_tool(ToolCapability('bellium/inpaint_knn', EscalationTier.KNN_EXEMPLAR, {'image'}, {'inpaint'}))
    router.register_tool(ToolCapability('large/vision_diffusion', EscalationTier.GENERAL_LARGE, {'image'}, {'generative'}))
    req = TaskRequirement('bench_req', 'image', {'cutout'}, hard_constraints={'zero_tokens'})
    mean_t, min_t, max_t, p95_t, ops = time_function(lambda: router.route_task(req), iterations=100)
    v = 'PASS_WITHIN_BUDGET' if p95_t <= 0.5 else 'OVER_BUDGET'
    results.append(BenchmarkMetric('bellium.routing', 'route_task (3 tiers + constraints)', 100, mean_t, min_t, max_t, p95_t, ops, 0.5, v))
    
    # 5. Vision anomaly detection benchmark (100x100 frame)
    frame = Image.new('RGB', (100, 100), (128, 128, 128))
    mean_t, min_t, max_t, p95_t, ops = time_function(lambda: detect_visual_anomalies(frame), iterations=50)
    v = 'PASS_WITHIN_BUDGET' if p95_t <= 5.0 else 'OVER_BUDGET'
    results.append(BenchmarkMetric('bellium.vision', 'detect_visual_anomalies (100x100)', 50, mean_t, min_t, max_t, p95_t, ops, 5.0, v))
    
    # 6. Audio VAD benchmark (16000 samples = 1 sec buffer)
    pcm = [0.05 * math.sin(2 * math.pi * 300 * (i / 16000.0)) for i in range(16000)]
    mean_t, min_t, max_t, p95_t, ops = time_function(lambda: detect_voice_activity(pcm, sample_rate=16000), iterations=25)
    v = 'PASS_WITHIN_BUDGET' if p95_t <= 15.0 else 'OVER_BUDGET'
    results.append(BenchmarkMetric('bellium.audio', 'detect_voice_activity (1.0s PCM)', 25, mean_t, min_t, max_t, p95_t, ops, 15.0, v))
    
    # 7. JSON Adapter repair and schema enforcement benchmark
    schema = {
        'label': SchemaRule(expected_type=str, required=True, allowed_values={'player', 'npc'}),
        'confidence': SchemaRule(expected_type=float, required=True, min_value=0.0, max_value=1.0),
        'count': SchemaRule(expected_type=int, required=False, default=1),
    }
    dirty_json = '```json\n{\"label\": \"npc\", \"confidence\": \"0.94\",}\n```'
    mean_t, min_t, max_t, p95_t, ops = time_function(lambda: repair_and_validate_json(dirty_json, schema), iterations=100)
    v = 'PASS_WITHIN_BUDGET' if p95_t <= 1.0 else 'OVER_BUDGET'
    results.append(BenchmarkMetric('bellium.adapters', 'repair_and_validate_json (fence+strip+coerce)', 100, mean_t, min_t, max_t, p95_t, ops, 1.0, v))
    
    # Print report
    print(f'{"Module / Specialist Operation":<42} | {"Mean":<8} | {"p95":<8} | {"Ops/sec":<10} | {"Budget":<8} | Status')
    print('-' * 75)
    all_passed = True
    for m in results:
        status_str = '[PASS]' if m.verdict == 'PASS_WITHIN_BUDGET' else '[WARN]'
        if m.verdict != 'PASS_WITHIN_BUDGET':
            all_passed = False
        print(f'{m.operation:<42} | {m.mean_ms:6.2f}ms | {m.p95_ms:6.2f}ms | {m.ops_per_sec:8.1f}/s | {m.budget_ms:6.2f}ms | {status_str}')
        
    print('=' * 75)
    if all_passed:
        print('All 7 specialist benchmarks respected latency budgets without external GPU.')
    else:
        print('Some benchmarks exceeded target latency budget.')
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(run_benchmarks())
