from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch


def _force_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@dataclass
class ClothingConfig:
    # sampling
    frames_per_slide_max: int = 4        # how many frames per slide to classify
    min_face_conf: float = 0.55          # only trust face bbox above this

    # torso crop from face bbox
    torso_scale_w: float = 2.2           # torso width ~= face_w * this
    torso_scale_h: float = 3.2           # torso height ~= face_h * this
    torso_shift_y: float = 1.15          # move down from face center

    # read performance
    read_every_nth: int = 1              # keep 1 unless you want to skip

    # safety
    max_total_frames: int = 300          # hard cap to avoid runaway compute


def _clamp_box(x1, y1, x2, y2, w, h):
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(x1 + 1, min(int(x2), w))
    y2 = max(y1 + 1, min(int(y2), h))
    return x1, y1, x2, y2


def torso_crop_from_face_bbox(frame_bgr: np.ndarray, face_bbox: List[int], cfg: ClothingConfig) -> np.ndarray:
    """
    Make an upper-body / torso crop derived from face bbox.
    frame_bgr: HxWx3
    face_bbox: [x1,y1,x2,y2] in original frame coords
    """
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = face_bbox
    fw = max(1, x2 - x1)
    fh = max(1, y2 - y1)

    cx = x1 + fw / 2.0
    cy = y1 + fh / 2.0

    torso_w = fw * cfg.torso_scale_w
    torso_h = fh * cfg.torso_scale_h

    # shift down to include chest/torso
    torso_cy = cy + fh * cfg.torso_shift_y

    tx1 = cx - torso_w / 2.0
    ty1 = torso_cy - torso_h / 2.0
    tx2 = cx + torso_w / 2.0
    ty2 = torso_cy + torso_h / 2.0

    tx1, ty1, tx2, ty2 = _clamp_box(tx1, ty1, tx2, ty2, w, h)
    return frame_bgr[ty1:ty2, tx1:tx2]


def pick_best_frames_per_slide(
    slide_frame_mapping: Dict[int, Dict[str, Any]],
    face_crops_cache: Dict[int, List[Dict[str, Any]]],
    cfg: ClothingConfig,
) -> List[Dict[str, Any]]:
    """
    Returns a list of {slide_id, frame_idx, face_conf, bbox}.
    Picks top-N frames per slide by face confidence.
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
            if conf >= cfg.min_face_conf:
                candidates.append((conf, idx, best.get("bbox")))

        candidates.sort(reverse=True, key=lambda x: x[0])
        for conf, idx, bbox in candidates[: cfg.frames_per_slide_max]:
            if bbox is None:
                continue
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


def analyze_clothing(
    video_path: str,
    slide_frame_mapping: Dict[int, Dict[str, Any]],
    face_crops_cache: Dict[int, List[Dict[str, Any]]],
    clothing_classifier,
    *,
    cfg: Optional[ClothingConfig] = None,
) -> Dict[str, Any]:
    """
    clothing_classifier must implement:
      assess_appearance(list_of_frames_bgr_or_rgb, return_full=True) -> dict
    We will pass RGB crops (recommended for CLIP).
    """
    cfg = cfg or ClothingConfig()

    picked = pick_best_frames_per_slide(slide_frame_mapping, face_crops_cache, cfg)
    if not picked:
        return {
            "is_appropriate": None,
            "detected_attributes": [],
            "recommendation": "No suitable frames with faces found for clothing analysis.",
            "coverage": {"slides_with_samples": 0, "frames_used": 0},
            "per_slide": {},
        }

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    per_slide_frames_rgb: Dict[int, List[np.ndarray]] = {}
    per_slide_meta: Dict[int, List[Dict[str, Any]]] = {}

    def read_frame(idx: int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        return ok, frame

    for j, rec in enumerate(picked):
        if cfg.read_every_nth > 1 and (j % cfg.read_every_nth != 0):
            continue

        ok, frame_bgr = read_frame(rec["frame_idx"])
        if not ok or frame_bgr is None:
            continue

        torso_bgr = torso_crop_from_face_bbox(frame_bgr, rec["bbox"], cfg)
        torso_rgb = cv2.cvtColor(torso_bgr, cv2.COLOR_BGR2RGB)

        sid = rec["slide_id"]
        per_slide_frames_rgb.setdefault(sid, []).append(torso_rgb)
        per_slide_meta.setdefault(sid, []).append(rec)

    cap.release()
    _force_cleanup()

    # Flatten crops for classifier
    all_crops = []
    crop_owner = []
    for sid, crops in per_slide_frames_rgb.items():
        for c in crops:
            all_crops.append(c)
            crop_owner.append(sid)

    if not all_crops:
        return {
            "is_appropriate": None,
            "detected_attributes": [],
            "recommendation": "Could not decode any sampled frames for clothing analysis.",
            "coverage": {"slides_with_samples": 0, "frames_used": 0},
            "per_slide": {},
        }

    # Run classifier once for all crops
    out = clothing_classifier.assess_appearance(all_crops, return_full=True)

    # Build slide coverage info
    slides_with_samples = len(per_slide_frames_rgb)
    per_slide = {
        str(sid): {
            "frames_used": len(per_slide_frames_rgb[sid]),
            "best_face_conf": max([m["face_conf"] for m in per_slide_meta.get(sid, [])], default=0.0),
        }
        for sid in per_slide_frames_rgb.keys()
    }

    return {
        "is_appropriate": out.get("is_appropriate"),
        "detected_attributes": out.get("captions", []),
        "recommendation": out.get("recommendation", ""),
        "coverage": {
            "slides_with_samples": slides_with_samples,
            "frames_used": len(all_crops),
        },
        "per_slide": per_slide,
        "debug": {
            "picked_frames": picked[:50],  # avoid huge JSON
        }
    }
