from __future__ import annotations

import gc
import torch
from typing import Any, Dict, List, Optional

from video_analysis.frame_sampling_face_cache import build_face_cache, FaceCacheConfig


def build_face_cache_tool(
    video_path: str,
    segments: List[Dict[str, Any]],
    fps: Optional[float] = None,
    per_slide_frames: int = 12,
    batch_size: int = 24,
) -> Dict[str, Any]:
    
    cfg = FaceCacheConfig(
        per_slide_frames=per_slide_frames,
        batch_size=batch_size,
    )
    out = build_face_cache(video_path, segments, fps=fps, config=cfg)

    # Cleanup for agent safety
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return out
