# agents/tools/gaze_tool.py
from __future__ import annotations

import gc
import torch
from typing import Dict, Any, Optional

from speaker_feedback.video_analysis.gaze_analysis import analyze_gaze, GazeConfig


def gaze_analysis_tool(
    video_path: str,
    slide_frame_mapping: Dict[int, Dict[str, Any]],
    face_crops_cache: Dict[int, Any],
    gaze_estimator,
    idx_to_slide: Optional[Dict[int, Dict[str, Any]]] = None,
    frames_per_slide_max: int = 12,
    min_face_conf: float = 0.55,
    min_gaze_conf: float = 0.2,
    min_valid_frames_per_slide: int = 3,
    min_coverage_ratio: float = 0.2,
    min_overall_valid_ratio: float = 0.4,
    expand_scale: float = 1.3,
    min_face_size: int = 48,
) -> Dict[str, Any]:
    """
    Agent-facing wrapper for gaze analysis.

    gaze_estimator must implement:
      - process_frame(face_crop_bgr) -> (gaze_label, confidence)
      - summarize_gaze(list_of_labels) -> dict
    """

    cfg = GazeConfig(
        frames_per_slide_max=frames_per_slide_max,
        min_face_conf=min_face_conf,
        min_gaze_conf=min_gaze_conf,
        min_valid_frames_per_slide=min_valid_frames_per_slide,
        min_coverage_ratio=min_coverage_ratio,
        min_overall_valid_ratio=min_overall_valid_ratio,
        expand_scale=expand_scale,
        min_face_size=min_face_size,
    )

    out = analyze_gaze(
        video_path=video_path,
        slide_frame_mapping=slide_frame_mapping,
        face_crops_cache=face_crops_cache,
        gaze_estimator=gaze_estimator,
        idx_to_slide=idx_to_slide,
        cfg=cfg,
    )

    # ---- safety cleanup (critical in agent / notebook) ----
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return out
