from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN


def _force_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_video_fps(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if not fps or fps <= 1e-6:
        fps = 25.0
    return float(fps)


def time_to_frame_idx(t_sec: float, fps: float) -> int:
    return int(round(float(t_sec) * float(fps)))


def sample_frame_indices_for_slide(
    start_t: float,
    end_t: float,
    fps: float,
    *,
    per_slide: int = 12,
    edge_pad_sec: float = 0.2,
    edge_pad_ratio: float = 0.05,
) -> List[int]:
    """Uniform sampling inside [start_t, end_t], avoiding edges."""
    start_t = float(start_t)
    end_t = float(end_t)

    if end_t <= start_t:
        return [time_to_frame_idx(start_t, fps)]

    pad = min(edge_pad_sec, (end_t - start_t) * edge_pad_ratio)
    a = start_t + pad
    b = end_t - pad
    if b <= a:
        a, b = start_t, end_t

    if per_slide <= 1:
        return [time_to_frame_idx((a + b) / 2.0, fps)]

    times = np.linspace(a, b, per_slide)
    idxs = [time_to_frame_idx(t, fps) for t in times]

    # dedup, keep order
    out: List[int] = []
    seen = set()
    for x in idxs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


@dataclass
class FaceCacheConfig:
    per_slide_frames: int = 12
    batch_size: int = 24
    resize_max_width: int = 640
    min_face_size: int = 20
    prob_thresh: float = 0.5
    mtcnn_thresholds: Tuple[float, float, float] = (0.3, 0.4, 0.5)
    mtcnn_factor: float = 0.7
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def build_slide_frame_mapping(
    segments: List[Dict[str, Any]], fps: float, per_slide_frames: int
) -> Dict[int, Dict[str, Any]]:
    slide_frame_mapping: Dict[int, Dict[str, Any]] = {}

    for seg in segments:
        sid = int(seg["slide_id"])
        st = float(seg["start_time"])
        et = float(seg["end_time"])
        idxs = sample_frame_indices_for_slide(st, et, fps, per_slide=per_slide_frames)

        slide_frame_mapping[sid] = {
            "frame_indices": idxs,
            "start_time": st,
            "end_time": et,
            "face_count_per_frame": {},
            "frames_with_faces": 0,
            "frames_without_faces": 0,
            # filled later:
            "face_detection_rate": 0.0,
        }

    return slide_frame_mapping


def build_face_cache(
    video_path: str,
    segments: List[Dict[str, Any]],
    *,
    fps: Optional[float] = None,
    config: Optional[FaceCacheConfig] = None,
) -> Dict[str, Any]:
    """
    Returns:
      {
        "fps": float,
        "slide_frame_mapping": { slide_id: {...} },
        "face_crops_cache": { frame_idx: [ {bbox:[x1,y1,x2,y2], confidence:float, area:float} ] },
        "stats": {...}
      }
    """
    cfg = config or FaceCacheConfig()
    fps_val = float(fps or get_video_fps(video_path))

    slide_frame_mapping = build_slide_frame_mapping(segments, fps_val, cfg.per_slide_frames)

    # flatten indices
    all_frame_indices: List[int] = []
    frame_to_slide: Dict[int, int] = {}
    for sid, m in slide_frame_mapping.items():
        for idx in m["frame_indices"]:
            all_frame_indices.append(idx)
            # IMPORTANT: prevent overwriting if two slides share the same rounded frame index
            if idx not in frame_to_slide:
                frame_to_slide[idx] = sid

    all_frame_indices = sorted(set(all_frame_indices))

    mtcnn = MTCNN(
        keep_all=True,
        post_process=False,
        min_face_size=cfg.min_face_size,
        thresholds=list(cfg.mtcnn_thresholds),
        factor=cfg.mtcnn_factor,
        device=cfg.device,
    )

    face_crops_cache: Dict[int, List[Dict[str, Any]]] = {}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    t0 = time.time()
    total_processed = 0

    def read_frame_by_idx(idx: int):
        # clamp idx defensively
        if frame_count > 0:
            idx = max(0, min(frame_count - 1, int(idx)))
        else:
            idx = int(idx)

        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        return ok, frame

    def box_area_xyxy(b: np.ndarray) -> float:
        x1, y1, x2, y2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    for i in range(0, len(all_frame_indices), cfg.batch_size):
        batch_idxs = all_frame_indices[i : i + cfg.batch_size]
        batch_rgb: List[np.ndarray] = []
        valid_idxs: List[int] = []
        scales: Dict[int, float] = {}

        for idx in batch_idxs:
            ok, frame = read_frame_by_idx(idx)
            if not ok or frame is None:
                continue

            h, w = frame.shape[:2]
            if w > cfg.resize_max_width:
                scale = cfg.resize_max_width / float(w)
                frame_small = cv2.resize(
                    frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
                )
            else:
                scale = 1.0
                frame_small = frame

            rgb_small = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
            batch_rgb.append(rgb_small)
            valid_idxs.append(idx)
            scales[idx] = scale

        if not batch_rgb:
            continue

        try:
            boxes_list, probs_list = mtcnn.detect(batch_rgb, landmarks=False)
        except Exception:
            boxes_list = [None] * len(valid_idxs)
            probs_list = [None] * len(valid_idxs)

        for idx, boxes, probs in zip(valid_idxs, boxes_list, probs_list):
            sid = frame_to_slide.get(idx)
            if sid is None:
                continue

            best = None
            if boxes is not None and probs is not None:
                candidates = [
                    (b, p)
                    for b, p in zip(boxes, probs)
                    if p is not None and float(p) >= cfg.prob_thresh
                ]
                if candidates:
                    # pick the largest face (more reliable than max prob)
                    b, p = max(candidates, key=lambda x: box_area_xyxy(x[0]))
                    scale = scales[idx]
                    x1, y1, x2, y2 = [int(coord / scale) for coord in b[:4]]
                    best = {
                        "bbox": [x1, y1, x2, y2],
                        "confidence": float(p),
                        "area": float(box_area_xyxy(b) / (scale * scale)),
                    }

            if best is not None:
                face_crops_cache[idx] = [best]
                slide_frame_mapping[sid]["face_count_per_frame"][idx] = 1
                slide_frame_mapping[sid]["frames_with_faces"] += 1
            else:
                slide_frame_mapping[sid]["face_count_per_frame"][idx] = 0
                slide_frame_mapping[sid]["frames_without_faces"] += 1

            total_processed += 1

        if (i // cfg.batch_size) % 4 == 0:
            _force_cleanup()

    cap.release()
    _force_cleanup()

    # finalize per-slide detection rates
    for sid, m in slide_frame_mapping.items():
        total = int(m["frames_with_faces"]) + int(m["frames_without_faces"])
        m["face_detection_rate"] = (float(m["frames_with_faces"]) / total) if total else 0.0

    t1 = time.time()
    stats = {
        "total_sampled_frames": len(all_frame_indices),
        "processed_frames": total_processed,
        "frames_with_faces": len(face_crops_cache),
        "face_detection_rate": (len(face_crops_cache) / len(all_frame_indices)) if all_frame_indices else 0.0,
        "time_sec": t1 - t0,
    }

    return {
        "fps": fps_val,
        "slide_frame_mapping": slide_frame_mapping,
        "face_crops_cache": face_crops_cache,
        "stats": stats,
    }
