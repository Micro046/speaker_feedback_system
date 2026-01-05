from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from collections import defaultdict, Counter

import cv2
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
    top_evidence_frames: int = 3  # store best frames per slide for NAT citations


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


def pick_frames(slide_frame_mapping, face_crops_cache, cfg: GazeConfig):
    """
    Pick up to N frames per slide.
    Prefer largest face area if available, else fallback to confidence.
    """
    picked = []
    for sid, info in slide_frame_mapping.items():
        cands = []
        for idx in info.get("frame_indices", []):
            faces = face_crops_cache.get(idx)
            if not faces:
                continue

            # prefer largest face if area exists
            if any("area" in f for f in faces):
                best = max(faces, key=lambda f: float(f.get("area", 0.0)))
            else:
                best = max(faces, key=lambda f: float(f.get("confidence", 0.0)))

            conf = float(best.get("confidence", 0.0))
            bbox = best.get("bbox")
            if bbox is None:
                continue
            if conf >= cfg.min_face_conf:
                area = float(best.get("area", 0.0))
                cands.append((area, conf, int(idx), bbox))

        # sort by area desc, then conf desc
        cands.sort(reverse=True, key=lambda x: (x[0], x[1]))
        for _, conf, idx, bbox in cands[: cfg.frames_per_slide_max]:
            picked.append({
                "slide_id": int(sid),
                "frame_idx": int(idx),
                "face_conf": float(conf),
                "bbox": bbox
            })

    picked.sort(key=lambda x: (x["slide_id"], -x["face_conf"]))
    if len(picked) > cfg.max_total_frames:
        picked = picked[: cfg.max_total_frames]
    return picked


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
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    # Store per-slide records with confidence and evidence
    slide_records: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    all_labels: List[str] = []

    for rec in picked:
        cap.set(cv2.CAP_PROP_POS_FRAMES, rec["frame_idx"])
        ok, frame = cap.read()
        if not ok or frame is None:
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

        conf = float(conf) if conf is not None else 0.0
        sid = int(rec["slide_id"])

        slide_records[sid].append({
            "frame_idx": int(rec["frame_idx"]),
            "t_sec": float(rec["frame_idx"]) / float(fps),
            "gaze": gaze,
            "gaze_conf": conf,
            "face_conf": float(rec["face_conf"]),
        })

        all_labels.append(gaze if conf >= 0.2 else "no_gaze")  # light guard

    cap.release()
    _cleanup()

    # ---- overall summary ----
    overall = gaze_estimator.summarize_gaze(all_labels)

    slide_summaries: Dict[str, Any] = {}
    issues: Dict[str, Any] = {}

    for sid in sorted(slide_frame_mapping.keys()):
        sampled = len(slide_frame_mapping[sid].get("frame_indices", []))
        recs = slide_records.get(sid, [])

        valid = [r for r in recs if r["gaze"] in {"left", "right", "center"} and r["gaze_conf"] >= 0.2]
        valid_ratio = (len(valid) / sampled) if sampled else 0.0

        labels = [r["gaze"] for r in valid]
        summ = gaze_estimator.summarize_gaze(labels) if labels else {
            "most_common_valid": "no_gaze",
            "valid_gaze_ratio": 0.0,
            "percentages": {"left": 0.0, "right": 0.0, "center": 0.0},
            "recommendation": "Limited reliable gaze frames. Improve lighting and face visibility.",
        }

        # evidence frames: pick best confidence among valid frames
        evidence = sorted(valid, key=lambda r: r["gaze_conf"], reverse=True)[: cfg.top_evidence_frames]

        slide_summaries[str(sid)] = {
            "dominant_gaze": summ["most_common_valid"],
            "valid_gaze_ratio": float(round(valid_ratio, 3)),
            "distribution": summ["percentages"],
            "recommendation": summ["recommendation"],
            "sampled_frames": int(sampled),
            "valid_frames": int(len(valid)),
            "coverage_ratio": float(round(valid_ratio, 3)),
            "avg_gaze_confidence": float(round(sum(r["gaze_conf"] for r in valid) / max(len(valid), 1), 3)),
            "evidence_frames": evidence,  # includes frame_idx + t_sec
            "slide_content": idx_to_slide.get(int(sid), {}).get("slide_content", ""),
            "audio_content": idx_to_slide.get(int(sid), {}).get("audio_content", ""),
        }

        slide_issues = []
        if valid_ratio < 0.3:
            slide_issues.append("Low gaze coverage/confidence")
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
            "frames_selected": int(len(picked)),
            "slides_covered": int(len(slide_records)),
            "fps": float(fps),
            "config": {
                "frames_per_slide_max": cfg.frames_per_slide_max,
                "min_face_conf": cfg.min_face_conf,
                "expand_scale": cfg.expand_scale,
                "min_face_size": cfg.min_face_size,
            }
        },
    }
