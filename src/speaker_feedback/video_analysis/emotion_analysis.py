from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict, Counter

import cv2
import numpy as np
import torch


def _force_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@dataclass
class EmotionConfig:
    # sampling
    frames_per_slide_max: int = 12
    min_face_conf: float = 0.55

    # face crop settings
    expand_scale: float = 1.25
    min_face_size: int = 48

    # batching
    batch_size: int = 32

    # slide aggregation
    min_valid_frames_per_slide: int = 2

    # stability / cleanup
    max_total_frames: int = 400

    # evidence
    top_frames_per_slide: int = 3  # keep a few best frames for citations


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


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def _entropy(p: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1.0)
    return float(-np.sum(p * np.log(p)))


def pick_best_frames_per_slide(
    slide_frame_mapping: Dict[int, Dict[str, Any]],
    face_crops_cache: Dict[int, List[Dict[str, Any]]],
    cfg: EmotionConfig,
) -> List[Dict[str, Any]]:
    """
    Returns list of {slide_id, frame_idx, face_conf, bbox} chosen top-N per slide.
    Prefers largest face area (if available) to track speaker reliably.
    """
    picked: List[Dict[str, Any]] = []

    for slide_id, info in slide_frame_mapping.items():
        candidates = []
        for idx in info.get("frame_indices", []):
            faces = face_crops_cache.get(idx)
            if not faces:
                continue

            # prefer largest face if area exists, else fallback to confidence
            if any("area" in f for f in faces):
                best = max(faces, key=lambda d: float(d.get("area", 0.0)))
            else:
                best = max(faces, key=lambda d: float(d.get("confidence", 0.0)))

            conf = float(best.get("confidence", 0.0))
            bbox = best.get("bbox")
            if bbox is None:
                continue
            if conf >= cfg.min_face_conf:
                area = float(best.get("area", 0.0))
                candidates.append((area, conf, int(idx), bbox))

        # sort by area desc then confidence desc
        candidates.sort(reverse=True, key=lambda x: (x[0], x[1]))

        for _, conf, idx, bbox in candidates[: cfg.frames_per_slide_max]:
            picked.append({
                "slide_id": int(slide_id),
                "frame_idx": int(idx),
                "face_conf": float(conf),
                "bbox": bbox,
            })

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
        batch = face_crops_rgb[i:i + bs]
        features = fer.extract_features(batch)
        _, scores = fer.classify_emotions(features, logits=True)
        all_scores.extend(scores)

        if (i // bs) % 3 == 0:
            _force_cleanup()

    all_scores = np.array(all_scores)  # (N, num_emotions)
    if all_scores.ndim != 2 or all_scores.shape[0] != len(face_meta):
        raise RuntimeError("Emotion model returned unexpected score shape.")

    # If logits, compute probability distribution for entropy/margins
    probs = _softmax(all_scores)

    # ---- Overall stats ----
    avg_scores = np.mean(probs, axis=0)
    emotion_idx = int(np.argmax(avg_scores))
    most_common_emotion = fer.idx_to_emotion_class[emotion_idx]

    emotion_counts = defaultdict(int)
    for row in probs:
        eidx = int(np.argmax(row))
        emotion_counts[fer.idx_to_emotion_class[eidx]] += 1

    overall_stats = {
        "most_common_emotion": most_common_emotion,
        "confidence": float(np.max(avg_scores)),
        "emotion_distribution": dict(emotion_counts),
        "total_faces_analyzed": int(len(face_crops_rgb)),
        "average_scores": {fer.idx_to_emotion_class[i]: float(s) for i, s in enumerate(avg_scores)},
        "avg_entropy": float(np.mean([_entropy(p) for p in probs])),
    }

    # ---- Slide-level aggregation ----
    per_slide_probs = defaultdict(list)
    per_slide_frames = defaultdict(list)

    for meta, p in zip(face_meta, probs):
        sid = int(meta["slide_id"])
        eidx = int(np.argmax(p))
        # dominant margin = top1 - top2 (stability proxy)
        top2 = np.partition(p, -2)[-2:]
        margin = float(np.max(top2) - np.min(top2))

        per_slide_probs[sid].append(p)
        per_slide_frames[sid].append({
            "frame_idx": int(meta["frame_idx"]),
            "face_conf": float(meta["face_conf"]),
            "emotion": fer.idx_to_emotion_class[eidx],
            "confidence": float(np.max(p)),
            "entropy": _entropy(p),
            "dominant_margin": margin,
        })

    slide_summaries: Dict[str, Any] = {}

    for sid in sorted(slide_frame_mapping.keys()):
        sampled_total = len(slide_frame_mapping[sid].get("frame_indices", []))
        frames = per_slide_frames.get(sid, [])
        valid_faces = len(frames)
        coverage_ratio = (valid_faces / sampled_total) if sampled_total else 0.0

        if valid_faces < cfg.min_valid_frames_per_slide:
            slide_summaries[str(sid)] = {
                "dominant_emotion": "no_face",
                "emotion_frequency": 0.0,
                "emotion_distribution": {"no_face": valid_faces},
                "sampled_frames": int(sampled_total),
                "valid_faces": int(valid_faces),
                "coverage_ratio": float(coverage_ratio),
                "avg_confidence": 0.0,
                "avg_entropy": 0.0,
                "avg_dominant_margin": 0.0,
                "has_valid_emotions": False,
                "top_frames": [],
                "slide_content": idx_to_slide.get(int(sid), {}).get("slide_content", ""),
                "audio_content": idx_to_slide.get(int(sid), {}).get("audio_content", ""),
            }
            continue

        # distribution & dominant
        dist = Counter([f["emotion"] for f in frames])
        dominant = dist.most_common(1)[0][0]
        freq = dist[dominant] / max(1, sum(dist.values()))

        avg_conf = float(np.mean([f["confidence"] for f in frames])) if frames else 0.0
        avg_ent = float(np.mean([f["entropy"] for f in frames])) if frames else 0.0
        avg_margin = float(np.mean([f["dominant_margin"] for f in frames])) if frames else 0.0

        # evidence frames: top by confidence
        top_frames = sorted(frames, key=lambda r: r["confidence"], reverse=True)[: cfg.top_frames_per_slide]

        slide_summaries[str(sid)] = {
            "dominant_emotion": dominant,
            "emotion_frequency": float(freq),
            "emotion_distribution": dict(dist),
            "sampled_frames": int(sampled_total),
            "valid_faces": int(valid_faces),
            "coverage_ratio": float(round(coverage_ratio, 3)),
            "avg_confidence": float(avg_conf),
            "avg_entropy": float(avg_ent),
            "avg_dominant_margin": float(avg_margin),
            "has_valid_emotions": True,
            "top_frames": top_frames,  # includes frame_idx for citations
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
            "batch_size": cfg.batch_size,
        },
    }

    return {
        "overall_stats": overall_stats,
        "slide_summaries": slide_summaries,
        "meta": meta,
    }
