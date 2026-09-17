"""Lightweight voice activity detection (VAD) and audio quality assessment.

Operates on raw PCM / floating point audio buffers or WAV audio arrays using deterministic
short-time energy (STE), zero-crossing rate (ZCR), spectral flux proxies and clipping checks.
Zero external dependencies: pure Python math.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import List, Tuple, Sequence, Optional


@dataclass
class AudioSegment:
    start_ms: float
    end_ms: float
    is_speech: bool
    mean_energy_db: float
    confidence: float


@dataclass
class AudioQualityReport:
    sample_rate: int
    duration_ms: float
    speech_ratio: float
    snr_db_estimate: float
    clipping_ratio: float
    is_clipped: bool
    segments: List[AudioSegment] = field(default_factory=list)
    overall_quality: str = "good"  # 'good', 'noisy', 'clipped', 'silent'


class VoiceActivityDetector:
    """Deterministic energy and zero-crossing rate voice activity detector."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_duration_ms: float = 30.0,
        energy_threshold_db: float = -42.0,
        zcr_speech_min: float = 0.02,
        zcr_speech_max: float = 0.45,
    ):
        self.sample_rate = sample_rate
        self.frame_len = max(1, int(sample_rate * (frame_duration_ms / 1000.0)))
        self.energy_threshold_db = energy_threshold_db
        self.zcr_speech_min = zcr_speech_min
        self.zcr_speech_max = zcr_speech_max
        
    def process_samples(self, samples: Sequence[float]) -> AudioQualityReport:
        if not samples:
            return AudioQualityReport(self.sample_rate, 0.0, 0.0, 0.0, 0.0, False, [], 'silent')
            
        n = len(samples)
        duration_ms = (n / self.sample_rate) * 1000.0
        
        # Check clipping (samples hitting >= 0.99 or <= -0.99)
        clipped_count = sum(1 for s in samples if abs(s) >= 0.99)
        clipping_ratio = clipped_count / n
        is_clipped = clipping_ratio > 0.005
        
        frames: List[AudioSegment] = []
        speech_frame_count = 0
        noise_energies = []
        speech_energies = []
        
        num_frames = max(1, n // self.frame_len)
        for i in range(num_frames):
            start_idx = i * self.frame_len
            end_idx = min(n, start_idx + self.frame_len)
            frame = samples[start_idx:end_idx]
            flen = len(frame)
            if flen == 0:
                continue
                
            # 1. Short-Time Energy (STE) RMS
            energy = sum(x * x for x in frame) / flen
            rms = math.sqrt(energy)
            energy_db = 20.0 * math.log10(max(1e-7, rms))
            
            # 2. Zero-Crossing Rate (ZCR)
            zcr = sum(1 for j in range(1, flen) if (frame[j] >= 0) != (frame[j - 1] >= 0)) / flen
            
            # 3. Decision rule
            is_speech = (energy_db >= self.energy_threshold_db) and (self.zcr_speech_min <= zcr <= self.zcr_speech_max)
            
            if is_speech:
                speech_frame_count += 1
                speech_energies.append(energy_db)
                conf = min(1.0, 0.5 + (energy_db - self.energy_threshold_db) / 40.0)
            else:
                noise_energies.append(energy_db)
                conf = min(1.0, 0.5 + (self.energy_threshold_db - energy_db) / 40.0)
                
            start_ms = (start_idx / self.sample_rate) * 1000.0
            end_ms = (end_idx / self.sample_rate) * 1000.0
            frames.append(AudioSegment(round(start_ms, 1), round(end_ms, 1), is_speech, round(energy_db, 1), round(conf, 2)))
            
        speech_ratio = speech_frame_count / num_frames
        
        # SNR estimation
        avg_speech_db = (sum(speech_energies) / len(speech_energies)) if speech_energies else -60.0
        avg_noise_db = (sum(noise_energies) / len(noise_energies)) if noise_energies else -80.0
        snr_est = max(0.0, avg_speech_db - avg_noise_db)
        
        # Overall quality label
        if is_clipped:
            quality = "clipped"
        elif speech_ratio == 0.0:
            quality = "silent"
        elif snr_est < 8.0:
            quality = "noisy"
        else:
            quality = "good"
            
        return AudioQualityReport(
            sample_rate=self.sample_rate,
            duration_ms=round(duration_ms, 1),
            speech_ratio=round(speech_ratio, 3),
            snr_db_estimate=round(snr_est, 1),
            clipping_ratio=round(clipping_ratio, 4),
            is_clipped=is_clipped,
            segments=frames,
            overall_quality=quality,
        )


def detect_voice_activity(samples: Sequence[float], sample_rate: int = 16000, **kwargs) -> AudioQualityReport:
    vad = VoiceActivityDetector(sample_rate=sample_rate, **kwargs)
    return vad.process_samples(samples)
