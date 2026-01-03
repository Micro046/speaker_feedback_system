from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from collections import defaultdict, Counter

import cv2
import numpy as np
import torch


def _cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@dataclass
class GazeConfig:
    frames_per_slide_max: int = 6
    min_face_conf: float = 0.55
    expand_scale: float = 1.3
    min_face_size: int = 48
    max_total_frames: int = 300


def _clamp(x1, y1, x2, y2, w, h):
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(x1 + 1, min(int(x2), w))
    y2 = max(y1 + 1, min(int(y2), h))
    return x1, y1, x2, y2


def _expand_bbox(bbox, w, h, scale):
    x1, y1, x2, y2 = bbox
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    cx, cy = x1 + bw / 2, y1 + bh / 2
    nw, nh = bw * scale, bh * scale
    return _clamp(cx - nw / 2, cy - nh / 2, cx + nw / 2, cy + nh / 2, w, h)


def pick_frames(slide_frame_mapping, face_crops_cache, cfg):
    picked = []
    for sid, info in slide_frame_mapping.items():
        cands = []
        for idx in info["frame_indices"]:
            faces = face_crops_cache.get(idx)
            if not faces:
                continue
            best = max(faces, key=lambda f: f["confidence"])
            if best["confidence"] >= cfg.min_face_conf:
                cands.append((best["confidence"], idx, best["bbox"]))
        cands.sort(reverse=True)
        for conf, idx, bbox in cands[:cfg.frames_per_slide_max]:
            picked.append({
                "slide_id": sid,
                "frame_idx": idx,
                "face_conf": conf,
                "bbox": bbox
            })
    picked.sort(key=lambda x: (x["slide_id"], -x["face_conf"]))
    return picked[:cfg.max_total_frames]


def analyze_gaze(
    video_path: str,
    slide_frame_mapping: Dict[int, Dict[str, Any]],
    face_crops_cache: Dict[int, List[Dict[str, Any]]],
    gaze_estimator,
    idx_to_slide: Optional[Dict[int, Dict[str, Any]]] = None,
    cfg: Optional[GazeConfig] = None,
) -> Dict[str, Any]:

    cfg = cfg or GazeConfig()
    idx_to_slide = idx_to_slide or {}

    picked = pick_frames(slide_frame_mapping, face_crops_cache, cfg)
    if not picked:
        return {"overall_summary": {}, "slide_summaries": {}, "issues": {}, "meta": {}}

    cap = cv2.VideoCapture(video_path)
    slide_gazes = defaultdict(list)
    all_gazes = []

    for rec in picked:
        cap.set(cv2.CAP_PROP_POS_FRAMES, rec["frame_idx"])
        ok, frame = cap.read()
        if not ok:
            continue

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = _expand_bbox(rec["bbox"], w, h, cfg.expand_scale)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0 or min(crop.shape[:2]) < cfg.min_face_size:
            continue

        try:
            gaze, conf = gaze_estimator.process_frame(crop)
        except Exception:
            gaze, conf = "no_gaze", 0.0

        slide_gazes[rec["slide_id"]].append(gaze)
        all_gazes.append(gaze)

    cap.release()
    _cleanup()

    overall = gaze_estimator.summarize_gaze(all_gazes)

    slide_summaries = {}
    issues = {}

    for sid, gazes in slide_gazes.items():
        summ = gaze_estimator.summarize_gaze(gazes)
        slide_summaries[str(sid)] = {
            "dominant_gaze": summ["most_common_valid"],
            "valid_gaze_ratio": summ["valid_gaze_ratio"],
            "distribution": summ["percentages"],
            "recommendation": summ["recommendation"],
            "slide_content": idx_to_slide.get(sid, {}).get("slide_content", ""),
        }

        slide_issues = []
        if summ["valid_gaze_ratio"] < 0.3:
            slide_issues.append("Low confidence eye contact")
        if summ["percentages"].get("left", 0) > 40:
            slide_issues.append("Excessive left gaze")
        if summ["percentages"].get("right", 0) > 40:
            slide_issues.append("Excessive right gaze")
        if slide_issues:
            issues[str(sid)] = slide_issues

    return {
        "overall_summary": overall,
        "slide_summaries": slide_summaries,
        "issues": issues,
        "meta": {
            "frames_used": len(picked),
            "slides_covered": len(slide_gazes),
        },
    }
