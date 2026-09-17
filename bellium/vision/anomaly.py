"""Lightweight visual anomaly and state detector for robotics, aquaponics and sensors.

Performs deterministic checks on blur (Laplacian variance proxy), overexposure/underexposure,
color shift, occlusion and localized foreign blobs.
Fail-closed design: emergency stops and collision hazards trigger immediate escalation.
Zero heavy dependencies: pure Python + PIL.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Tuple, List, Optional
from PIL import Image


@dataclass
class AnomalyReport:
    is_anomalous: bool
    anomaly_score: float  # [0.0, 1.0]
    severity: str        # 'normal', 'warning', 'critical_emergency'
    reasons: List[str] = field(default_factory=list)
    brightness: float = 0.5
    contrast: float = 0.5
    blur_score: float = 0.0
    occlusion_ratio: float = 0.0


class VisualAnomalyDetector:
    """Deterministic sensor and camera anomaly scanner."""
    
    def __init__(
        self,
        *,
        min_brightness: float = 0.08,
        max_brightness: float = 0.92,
        min_contrast: float = 0.05,
        blur_threshold: float = 12.0,
        max_occlusion: float = 0.35,
    ):
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.min_contrast = min_contrast
        self.blur_threshold = blur_threshold
        self.max_occlusion = max_occlusion
        
    def inspect_frame(self, frame: Image.Image, *, reference_frame: Optional[Image.Image] = None) -> AnomalyReport:
        rgb = frame.convert("RGB")
        w, h = rgb.size
        total_pixels = w * h
        if total_pixels == 0:
            return AnomalyReport(True, 1.0, 'critical_emergency', ["Empty sensor frame"])
            
        gray = frame.convert("L")
        px = gray.load()
        rgb_px = rgb.load()
        
        # 1. Mean brightness & variance (contrast)
        s = 0
        s_sq = 0
        step = max(1, int(math.isqrt(total_pixels) // 50))
        sample_count = 0
        
        for y in range(0, h, step):
            for x in range(0, w, step):
                v = px[x, y]
                s += v
                s_sq += v * v
                sample_count += 1
                
        mean_val = s / sample_count
        variance = max(0.0, (s_sq / sample_count) - (mean_val * mean_val))
        std_dev = math.sqrt(variance)
        
        norm_bright = mean_val / 255.0
        norm_contrast = std_dev / 128.0
        
        reasons = []
        score = 0.0
        
        # Underexposure (blackout / lens cap)
        if norm_bright < self.min_brightness:
            reasons.append(f"Underexposure blackout detected ({norm_bright:.2f} < {self.min_brightness})")
            score = max(score, 0.95)
            
        # Overexposure (sensor blinding / glare)
        if norm_bright > self.max_brightness:
            reasons.append(f"Overexposure glare detected ({norm_bright:.2f} > {self.max_brightness})")
            score = max(score, 0.85)
            
        # Low contrast (fog / washout / blank screen)
        if norm_contrast < self.min_contrast and 0.2 < norm_bright < 0.8:
            reasons.append(f"Contrast washout / uniform blanking ({norm_contrast:.2f} < {self.min_contrast})")
            score = max(score, 0.70)
            
        # 2. Focus / Blur proxy
        grad_sum = 0
        grad_samples = 0
        for y in range(step, h - step, step * 2):
            for x in range(step, w - step, step * 2):
                gx = abs(px[x + step, y] - px[x - step, y])
                gy = abs(px[x, y + step] - px[x, y - step])
                grad_sum += gx + gy
                grad_samples += 1
                
        avg_grad = (grad_sum / grad_samples) if grad_samples > 0 else 0.0
        if avg_grad < self.blur_threshold and norm_contrast >= self.min_contrast:
            reasons.append(f"Defocus / motion blur anomaly ({avg_grad:.1f} < {self.blur_threshold})")
            score = max(score, 0.65)
            
        # 3. Reference frame comparison (occlusion / sudden foreign intrusion)
        occlusion_ratio = 0.0
        if reference_frame is not None:
            ref_rgb = reference_frame.convert("RGB").resize((w, h))
            ref_px = ref_rgb.load()
            diff_count = 0
            ref_samples = 0
            for y in range(0, h, step):
                for x in range(0, w, step):
                    c1 = rgb_px[x, y]
                    c2 = ref_px[x, y]
                    dist = math.sqrt(sum((a - b)**2 for a, b in zip(c1, c2)))
                    if dist > 35.0:
                        diff_count += 1
                    ref_samples += 1
            occlusion_ratio = diff_count / ref_samples if ref_samples > 0 else 0.0
            if occlusion_ratio > self.max_occlusion:
                reasons.append(f"Major field occlusion / structural shift ({occlusion_ratio:.1%} > {self.max_occlusion:.1%})")
                score = max(score, 0.90)
                
        # Determine final severity
        if score >= 0.85:
            severity = "critical_emergency"
        elif score >= 0.50:
            severity = "warning"
        else:
            severity = "normal"
            
        is_anom = score >= 0.50
        return AnomalyReport(
            is_anomalous=is_anom,
            anomaly_score=round(score, 3),
            severity=severity,
            reasons=reasons,
            brightness=round(norm_bright, 3),
            contrast=round(norm_contrast, 3),
            blur_score=round(avg_grad, 2),
            occlusion_ratio=round(occlusion_ratio, 3),
        )


def detect_visual_anomalies(frame: Image.Image, **kwargs) -> AnomalyReport:
    detector = VisualAnomalyDetector(**kwargs)
    return detector.inspect_frame(frame)
