from __future__ import annotations

import gc
import torch
from typing import Any, Dict, Optional

from speaker_feedback.video_analysis.emotion_analysis import analyze_emotions, EmotionConfig


def emotion_analysis_tool(
    video_path: str,
    slide_frame_mapping: Dict[int, Dict[str, Any]],
    face_crops_cache: Dict[int, Any],
    fer,
    idx_to_slide: Optional[Dict[int, Dict[str, Any]]] = None,
    frames_per_slide_max: int = 12,
    min_face_conf: float = 0.55,
    min_valid_frames_per_slide: int = 2,
    min_coverage_ratio: float = 0.2,
    min_overall_valid_ratio: float = 0.3,
    expand_scale: float = 1.3,
    min_face_size: int = 48,
    batch_size: int = 32,
) -> Dict[str, Any]:
    cfg = EmotionConfig(
        frames_per_slide_max=frames_per_slide_max,
        min_face_conf=min_face_conf,
        min_valid_frames_per_slide=min_valid_frames_per_slide,
        expand_scale=expand_scale,
        min_face_size=min_face_size,
        batch_size=batch_size,
    )

    out = analyze_emotions(
        video_path=video_path,
        slide_frame_mapping=slide_frame_mapping,
        face_crops_cache=face_crops_cache,
        fer=fer,
        idx_to_slide=idx_to_slide,
        cfg=cfg,
    )

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return out
