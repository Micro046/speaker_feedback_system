from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from collections import defaultdict, Counter


def _force_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@dataclass
class EmotionConfig:
    # sampling
    frames_per_slide_max: int = 6
    min_face_conf: float = 0.55

    # face crop settings
    expand_scale: float = 1.25     # expand bbox a bit for robustness
    min_face_size: int = 48        # skip very small crops

    # batching
    batch_size: int = 32

    # slide aggregation
    min_valid_frames_for_slide: int = 1  # if no valid frames -> "no_face"

    # stability / cleanup
    max_total_frames: int = 400     # global cap to avoid runaway compute


def _clamp_box(x1, y1, x2, y2, w, h):
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(x1 + 1, min(int(x2), w))
    y2 = max(y1 + 1, min(int(y2), h))
    return x1, y1, x2, y2


def expand_bbox(bbox: List[int], frame_w: int, frame_h: int, scale: float) -> List[int]:
    x1, y1, x2, y2 = bbox
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0

    nbw = bw * scale
    nbh = bh * scale

    nx1 = cx - nbw / 2.0
    ny1 = cy - nbh / 2.0
    nx2 = cx + nbw / 2.0
    ny2 = cy + nbh / 2.0
    nx1, ny1, nx2, ny2 = _clamp_box(nx1, ny1, nx2, ny2, frame_w, frame_h)
    return [nx1, ny1, nx2, ny2]


def pick_best_frames_per_slide(
    slide_frame_mapping: Dict[int, Dict[str, Any]],
    face_crops_cache: Dict[int, List[Dict[str, Any]]],
    cfg: EmotionConfig,
) -> List[Dict[str, Any]]:
    """
    Returns list of {slide_id, frame_idx, face_conf, bbox} chosen top-N per slide.
    """
    picked: List[Dict[str, Any]] = []

    for slide_id, info in slide_frame_mapping.items():
        candidates = []
        for idx in info.get("frame_indices", []):
            faces = face_crops_cache.get(idx)
            if not faces:
                continue
            best = max(faces, key=lambda d: float(d.get("confidence", 0.0)))
            conf = float(best.get("confidence", 0.0))
            if conf >= cfg.min_face_conf and best.get("bbox") is not None:
                candidates.append((conf, int(idx), best["bbox"]))

        candidates.sort(reverse=True, key=lambda x: x[0])

        for conf, idx, bbox in candidates[: cfg.frames_per_slide_max]:
            picked.append({
                "slide_id": int(slide_id),
                "frame_idx": int(idx),
                "face_conf": float(conf),
                "bbox": bbox,
            })

    # global cap
    picked.sort(key=lambda r: (r["slide_id"], -r["face_conf"]))
    if len(picked) > cfg.max_total_frames:
        picked = picked[: cfg.max_total_frames]

    return picked


def analyze_emotions(
    video_path: str,
    slide_frame_mapping: Dict[int, Dict[str, Any]],
    face_crops_cache: Dict[int, List[Dict[str, Any]]],
    fer,  # EmotiEffLibRecognizer
    idx_to_slide: Optional[Dict[int, Dict[str, Any]]] = None,
    cfg: Optional[EmotionConfig] = None,
) -> Dict[str, Any]:
    cfg = cfg or EmotionConfig()
    idx_to_slide = idx_to_slide or {}

    picked = pick_best_frames_per_slide(slide_frame_mapping, face_crops_cache, cfg)
    if not picked:
        return {
            "overall_stats": {
                "most_common_emotion": "no_face",
                "confidence": 0.0,
                "emotion_distribution": {},
                "total_faces_analyzed": 0,
                "average_scores": {},
            },
            "slide_summaries": {},
            "meta": {
                "frames_used": 0,
                "slides_with_samples": 0,
                "note": "No suitable face frames found for emotion analysis.",
            },
        }

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    # Prepare batches of face crops (RGB)
    face_crops_rgb: List[np.ndarray] = []
    face_meta: List[Dict[str, Any]] = []

    def read_frame(idx: int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        return ok, frame

    for rec in picked:
        ok, frame_bgr = read_frame(rec["frame_idx"])
        if not ok or frame_bgr is None:
            continue

        h, w = frame_bgr.shape[:2]
        bbox = expand_bbox(rec["bbox"], w, h, cfg.expand_scale)
        x1, y1, x2, y2 = bbox
        crop_bgr = frame_bgr[y1:y2, x1:x2]
        if crop_bgr.size == 0:
            continue

        ch, cw = crop_bgr.shape[:2]
        if ch < cfg.min_face_size or cw < cfg.min_face_size:
            continue

        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

        face_crops_rgb.append(crop_rgb)
        face_meta.append({
            "slide_id": rec["slide_id"],
            "frame_idx": rec["frame_idx"],
            "face_conf": rec["face_conf"],
        })

    cap.release()
    _force_cleanup()

    if not face_crops_rgb:
        return {
            "overall_stats": {
                "most_common_emotion": "no_face",
                "confidence": 0.0,
                "emotion_distribution": {},
                "total_faces_analyzed": 0,
                "average_scores": {},
            },
            "slide_summaries": {},
            "meta": {
                "frames_used": 0,
                "slides_with_samples": 0,
                "note": "Could not crop any valid faces from sampled frames.",
            },
        }

    # ---- Run FER in batches ----
    all_scores = []
    bs = int(cfg.batch_size)

    for i in range(0, len(face_crops_rgb), bs):
        batch = face_crops_rgb[i:i+bs]
        features = fer.extract_features(batch)
        _, scores = fer.classify_emotions(features, logits=True)
        all_scores.extend(scores)

        if (i // bs) % 3 == 0:
            _force_cleanup()

    all_scores = np.array(all_scores)  # (N, num_emotions)

    # ---- Overall stats ----
    avg_scores = np.mean(all_scores, axis=0)
    emotion_idx = int(np.argmax(avg_scores))
    most_common_emotion = fer.idx_to_emotion_class[emotion_idx]

    emotion_counts = defaultdict(int)
    for row in all_scores:
        eidx = int(np.argmax(row))
        emotion_counts[fer.idx_to_emotion_class[eidx]] += 1

    overall_stats = {
        "most_common_emotion": most_common_emotion,
        "confidence": float(np.max(avg_scores)),
        "emotion_distribution": dict(emotion_counts),
        "total_faces_analyzed": int(len(face_crops_rgb)),
        "average_scores": {fer.idx_to_emotion_class[i]: float(s) for i, s in enumerate(avg_scores)},
    }

    # ---- Slide-level aggregation ----
    per_slide_scores = defaultdict(list)
    per_slide_frames = defaultdict(list)

    for meta, score in zip(face_meta, all_scores):
        sid = meta["slide_id"]
        per_slide_scores[sid].append(score)
        per_slide_frames[sid].append({
            "frame_idx": meta["frame_idx"],
            "face_conf": float(meta["face_conf"]),
            "emotion": fer.idx_to_emotion_class[int(np.argmax(score))],
            "confidence": float(np.max(score)),
        })

    slide_summaries = {}

    for sid in sorted(slide_frame_mapping.keys()):
        frames = per_slide_frames.get(sid, [])
        if len(frames) < cfg.min_valid_frames_for_slide:
            slide_summaries[str(sid)] = {
                "dominant_emotion": "no_face",
                "emotion_frequency": 0.0,
                "emotion_distribution": {"no_face": slide_frame_mapping[sid].get("frames_with_faces", 0)},
                "total_frames": int(slide_frame_mapping[sid].get("frames_with_faces", 0) + slide_frame_mapping[sid].get("frames_without_faces", 0)),
                "total_faces": 0,
                "avg_confidence": 0.0,
                "has_valid_emotions": False,
                "slide_content": idx_to_slide.get(int(sid), {}).get("slide_content", ""),
                "audio_content": idx_to_slide.get(int(sid), {}).get("audio_content", ""),
            }
            continue

        # distribution
        dist = Counter([f["emotion"] for f in frames])
        dominant = dist.most_common(1)[0][0]
        valid_total = sum(dist.values())
        freq = dist[dominant] / max(1, valid_total)
        avg_conf = float(np.mean([f["confidence"] for f in frames])) if frames else 0.0

        slide_summaries[str(sid)] = {
            "dominant_emotion": dominant,
            "emotion_frequency": float(freq),
            "emotion_distribution": dict(dist),
            "total_frames": int(len(frames)),
            "total_faces": int(len(frames)),  # 1 best face per sampled frame in your cache
            "avg_confidence": avg_conf,
            "has_valid_emotions": dominant != "no_face",
            "slide_content": idx_to_slide.get(int(sid), {}).get("slide_content", ""),
            "audio_content": idx_to_slide.get(int(sid), {}).get("audio_content", ""),
        }

    meta = {
        "frames_used": int(len(face_crops_rgb)),
        "slides_with_samples": int(len(per_slide_frames)),
        "sampling": {
            "frames_per_slide_max": cfg.frames_per_slide_max,
            "min_face_conf": cfg.min_face_conf,
            "expand_scale": cfg.expand_scale,
            "min_face_size": cfg.min_face_size,
        }
    }

    return {
        "overall_stats": overall_stats,
        "slide_summaries": slide_summaries,
        "meta": meta,
    }
