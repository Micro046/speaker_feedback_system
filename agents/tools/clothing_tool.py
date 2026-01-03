from __future__ import annotations

import gc
import torch
from typing import Any, Dict, Optional

from video_analysis.clothing_analysis import analyze_clothing, ClothingConfig


def clothing_analysis_tool(
    video_path: str,
    slide_frame_mapping: Dict[int, Dict[str, Any]],
    face_crops_cache: Dict[int, Any],
    clothing_classifier,
    frames_per_slide_max: int = 4,
    min_face_conf: float = 0.55,
) -> Dict[str, Any]:
    cfg = ClothingConfig(
        frames_per_slide_max=frames_per_slide_max,
        min_face_conf=min_face_conf,
    )

    out = analyze_clothing(
        video_path=video_path,
        slide_frame_mapping=slide_frame_mapping,
        face_crops_cache=face_crops_cache,
        clothing_classifier=clothing_classifier,
        cfg=cfg,
    )

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return out
