"""End-to-end asset preparation pipeline for StoryCore and game production.

Chains deterministic cutout, background normalization, defect detection and localized inpaint.
Zero heavy dependencies: executes with PIL and pure Python math.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any
from PIL import Image

from bellium.cutout import extract_foreground, normalize_background, CutoutResult, MaskMetrics
from bellium.inpaint import inpaint_patch_knn, InpaintResult
from bellium.routing import HybridRouter, ToolCapability, TaskRequirement, EscalationTier


@dataclass
class AssetSpec:
    target_width: int = 256
    target_height: int = 256
    bg_mode: str = "transparent"  # 'transparent', 'white', or 'color'
    custom_bg_color: Tuple[int, int, int] = (255, 255, 255)
    auto_inpaint_defects: bool = True
    tolerance: float = 30.0


@dataclass
class AssetPrepReport:
    output_image: Image.Image
    mask: Image.Image
    cutout_metrics: MaskMetrics
    inpaint_applied: bool
    inpaint_pixels_filled: int
    verdict: str  # 'ready_production', 'needs_review', 'escalate'
    confidence: float
    log: list[str] = field(default_factory=list)


class AssetPrepPipeline:
    """Deterministic pipeline preparing game/story assets with quality gating."""
    
    def __init__(self, spec: Optional[AssetSpec] = None):
        self.spec = spec or AssetSpec()
        self.router = HybridRouter()
        # Register local tools
        self.router.register_tool(ToolCapability(
            tool_id='bellium/cutout',
            tier=EscalationTier.DETERMINISTIC,
            modalities={'image'},
            tags={'cutout', 'alpha', 'segmentation'},
            latency_budget_ms=10.0,
            success_rate=0.98,
        ))
        self.router.register_tool(ToolCapability(
            tool_id='bellium/inpaint_knn',
            tier=EscalationTier.KNN_EXEMPLAR,
            modalities={'image'},
            tags={'inpaint', 'fill'},
            latency_budget_ms=25.0,
            success_rate=0.92,
        ))
        
    def process_asset(
        self,
        image: Image.Image,
        defect_mask: Optional[Image.Image] = None,
    ) -> AssetPrepReport:
        """Execute the full preparation chain on an input asset."""
        log = []
        working_im = image.copy()
        
        # Step 1: Inpaint defects if provided and requested
        inpaint_applied = False
        pixels_filled = 0
        inpaint_needs_escalation = False
        if defect_mask is not None and self.spec.auto_inpaint_defects:
            inpaint_res = inpaint_patch_knn(working_im, defect_mask)
            working_im = inpaint_res.image
            pixels_filled = inpaint_res.metrics.filled_pixels
            inpaint_applied = pixels_filled > 0
            inpaint_needs_escalation = inpaint_res.metrics.verdict.method != "patch_knn"
            if inpaint_needs_escalation:
                log.append(f"Defect inpainting skipped: {inpaint_res.metrics.verdict.reason}")
            else:
                log.append(f"Defect inpainting applied: {pixels_filled} pixels repaired")
            
        # Step 2: Foreground cutout
        cutout_res = extract_foreground(working_im, tolerance=self.spec.tolerance)
        m = cutout_res.metrics
        log.append(f"Cutout: coverage={m.coverage_ratio:.2%}, uncertainty={m.edge_uncertainty:.2%}, bleed={m.border_bleed_ratio:.2%}")
        
        # Step 3: Background normalization or transparent placement
        if self.spec.bg_mode == "transparent":
            # Return transparent RGBA centered in target canvas
            fg_rgba = cutout_res.image
            bbox = m.bbox
            min_x, min_y, max_x, max_y = bbox
            fg_w = max(1, max_x - min_x + 1)
            fg_h = max(1, max_y - min_y + 1)
            fg_crop = fg_rgba.crop((min_x, min_y, max_x + 1, max_y + 1))
            
            # Create target RGBA canvas
            tw, th = self.spec.target_width, self.spec.target_height
            # Scale keeping aspect ratio to fit with 5% margin
            scale = min((tw * 0.9) / fg_w, (th * 0.9) / fg_h)
            new_w = max(1, int(fg_w * scale))
            new_h = max(1, int(fg_h * scale))
            fg_resized = fg_crop.resize((new_w, new_h), Image.Resampling.BILINEAR)
            
            out_canvas = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
            paste_x = (tw - new_w) // 2
            paste_y = (th - new_h) // 2
            out_canvas.paste(fg_resized, (paste_x, paste_y), fg_resized)
            output_im = out_canvas
            log.append(f"Formatted transparent asset scaled to {new_w}x{new_h} in {tw}x{th} canvas")
        else:
            # Solid background mode
            bg_col = (255, 255, 255) if self.spec.bg_mode == "white" else self.spec.custom_bg_color
            output_im = normalize_background(working_im, target_bg=bg_col)
            tw, th = self.spec.target_width, self.spec.target_height
            output_im = output_im.resize((tw, th), Image.Resampling.BILINEAR)
            log.append(f"Formatted solid background asset into {tw}x{th} canvas")
            
        # Final verdict determination
        if inpaint_needs_escalation:
            verdict = "escalate"
        elif m.recommendation == "confident":
            verdict = "ready_production"
        elif m.recommendation == "review":
            verdict = "needs_review"
        else:
            verdict = "escalate"
            
        return AssetPrepReport(
            output_image=output_im,
            mask=cutout_res.mask,
            cutout_metrics=m,
            inpaint_applied=inpaint_applied,
            inpaint_pixels_filled=pixels_filled,
            verdict=verdict,
            confidence=0.0 if inpaint_needs_escalation else m.confidence,
            log=log,
        )
