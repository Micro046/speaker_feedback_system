from __future__ import annotations

import gc
import torch
from typing import Any, Dict, Optional

from video_analysis.emotion_analysis import analyze_emotions, EmotionConfig


def emotion_analysis_tool(
    video_path: str,
    slide_frame_mapping: Dict[int, Dict[str, Any]],
    face_crops_cache: Dict[int, Any],
    fer,
    idx_to_slide: Optional[Dict[int, Dict[str, Any]]] = None,
    frames_per_slide_max: int = 6,
    min_face_conf: float = 0.55,
) -> Dict[str, Any]:
    cfg = EmotionConfig(
        frames_per_slide_max=frames_per_slide_max,
        min_face_conf=min_face_conf,
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
