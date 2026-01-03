# agents/tools/gesture_tool.py
from __future__ import annotations

import gc
import torch
from typing import Any, Dict, Optional

from video_analysis.gesture_analysis import (
    analyze_gestures_from_video,
    GestureConfig,
)


def gestures_analysis_tool(
    video_path: str,
    slide_frame_mapping: Dict[int, Dict[str, Any]],
    gesture_detector,
    idx_to_slide: Optional[Dict[int, Dict[str, Any]]] = None,
    *,
    frames_per_slide_max: int = 6,
    resize_w: int = 640,
    max_total_frames: int = 350,
) -> Dict[str, Any]:
    """
    Tool wrapper for gesture / body-language analysis.

    Inputs:
      - video_path: path to presentation video
      - slide_frame_mapping: output from face cache tool
      - gesture_detector: instance of Gestures (YOLOv8 pose)
      - idx_to_slide: optional metadata (slide_content, audio_content)

    Returns:
      {
        overall: {
          joint_statistics,
          recommendations,
          frames_processed
        },
        slide_summaries: {
          slide_id: {
            joint_statistics,
            detection_rate,
            slide_content
          }
        },
        issues: {
          slide_id: [issues...]
        },
        meta: {
          frames_used,
          slides_covered,
          frames_per_slide_max
        }
      }
    """

    cfg = GestureConfig(
        frames_per_slide_max=frames_per_slide_max,
        resize_w=resize_w,
        max_total_frames=max_total_frames,
    )

    out = analyze_gestures_from_video(
        video_path=video_path,
        slide_frame_mapping=slide_frame_mapping,
        gesture_detector=gesture_detector,
        idx_to_slide=idx_to_slide,
        cfg=cfg,
    )

    # aggressive cleanup for agent safety
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return out


def gesture_analysis_tool(
    video_path: str,
    slide_frame_mapping: Dict[int, Dict[str, Any]],
    gesture_detector,
    idx_to_slide: Optional[Dict[int, Dict[str, Any]]] = None,
    *,
    frames_per_slide_max: int = 6,
    resize_w: int = 640,
    max_total_frames: int = 350,
) -> Dict[str, Any]:
    """
    Backwards-compatible alias for notebooks or older imports.
    """
    return gestures_analysis_tool(
        video_path=video_path,
        slide_frame_mapping=slide_frame_mapping,
        gesture_detector=gesture_detector,
        idx_to_slide=idx_to_slide,
        frames_per_slide_max=frames_per_slide_max,
        resize_w=resize_w,
        max_total_frames=max_total_frames,
    )
