from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from scipy.spatial.distance import cosine


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
        # face_crops_cache: { frame_idx: [ {bbox, confidence, area, embedding} ] }
        "dominant_embedding": List[float],    
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
            if idx not in frame_to_slide:
                frame_to_slide[idx] = sid

    all_frame_indices = sorted(set(all_frame_indices))

    # 1. Models
    mtcnn = MTCNN(
        keep_all=True,
        post_process=False,
        min_face_size=cfg.min_face_size,
        thresholds=list(cfg.mtcnn_thresholds),
        factor=cfg.mtcnn_factor,
        device=cfg.device,
    )
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(cfg.device)

    face_crops_cache: Dict[int, List[Dict[str, Any]]] = {}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    t0 = time.time()
    total_processed = 0
    all_embeddings_list: List[np.ndarray] = []

    def read_frame_by_idx(idx: int):
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

    # 2. Batch Processing
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

        # Temp batch storage to run Resnet in batch? 
        # Actually Resnet expects 160x160 tensors. 
        # Let's extract crops, resize, and run batch.
        
        crops_in_batch = []
        metadata_in_batch = [] # (idx, bbox, etc)

        for idx, boxes, probs, img_rgb in zip(valid_idxs, boxes_list, probs_list, batch_rgb):
            sid = frame_to_slide.get(idx)
            if sid is None: continue

            if boxes is not None and probs is not None:
                candidates = [
                    (b, p) for b, p in zip(boxes, probs)
                    if p is not None and float(p) >= cfg.prob_thresh
                ]
                if candidates:
                    # Best face only
                    b, p = max(candidates, key=lambda x: box_area_xyxy(x[0]))
                    scale = scales[idx]
                    x1, y1, x2, y2 = [int(coord / scale) for coord in b[:4]]
                    
                    # Crop from ORIGINAL SMALL RGB (scaled) to save IO, then resize to 160x160
                    # boxes are in small coordinates
                    bx1, by1, bx2, by2 = [int(c) for c in b[:4]]
                    # Clamp
                    h_s, w_s = img_rgb.shape[:2]
                    bx1, by1 = max(0, bx1), max(0, by1)
                    bx2, by2 = min(w_s, bx2), min(h_s, by2)
                    
                    if bx2 > bx1 and by2 > by1:
                        face_img = img_rgb[by1:by2, bx1:bx2]
                        face_img = cv2.resize(face_img, (160, 160))
                        
                        # Normalize for InceptionResnetV1 (whitening)
                        face_tensor = torch.tensor(face_img).permute(2, 0, 1).float()
                        face_tensor = (face_tensor - 127.5) / 128.0
                        
                        crops_in_batch.append(face_tensor)
                        metadata_in_batch.append({
                            "frame_idx": idx,
                            "bbox": [x1, y1, x2, y2],
                            "confidence": float(p),
                            "area": float(box_area_xyxy(b) / (scale * scale)),
                            "slide_id": sid
                        })
            
            total_processed += 1

        # Run Resnet Batch
        if crops_in_batch:
            batch_t = torch.stack(crops_in_batch).to(cfg.device)
            with torch.no_grad():
                emb_batch = resnet(batch_t).cpu().numpy() # [N, 512]
            
            for meta, emb in zip(metadata_in_batch, emb_batch):
                # normalize embedding
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm
                
                meta["embedding"] = emb.tolist()
                
                # Store tentatively (will filter later)
                # Note: We store ALL processed faces. 
                # But we only picked ONE per frame above.
                idx = meta["frame_idx"]
                face_crops_cache[idx] = [meta]
                all_embeddings_list.append(emb)

        if (i // cfg.batch_size) % 4 == 0:
            _force_cleanup()

    cap.release()
    _force_cleanup()

    # 3. Clustering & Filtering
    # "Robust Dominant Speaker": Mean of all faces.
    # Refinement: Reject outliers iteratively? 
    # One-pass: Calc mean, keep faces with sim > 0.6.
    
    dominant_emb = None
    if all_embeddings_list:
        matrix = np.stack(all_embeddings_list)
        mean_vec = np.mean(matrix, axis=0)
        norm = np.linalg.norm(mean_vec)
        if norm > 0:
           dominant_emb = mean_vec / norm
        else:
           dominant_emb = mean_vec
    
    filtered_cache = {}
    filtered_count = 0
    
    threshold = 0.6
    
    if dominant_emb is not None:
        for idx, faces in face_crops_cache.items():
            valid_faces = []
            for f in faces:
                emb = np.array(f["embedding"])
                sim = np.dot(emb, dominant_emb)
                if sim >= threshold:
                    valid_faces.append(f)
            
            if valid_faces:
                filtered_cache[idx] = valid_faces
                filtered_count += len(valid_faces)
                
                # Update map
                sid = frame_to_slide[idx]
                slide_frame_mapping[sid]["face_count_per_frame"][idx] = len(valid_faces)
                slide_frame_mapping[sid]["frames_with_faces"] += 1
            else:
                # Face was detected but filtered out (not speaker)
                sid = frame_to_slide[idx]
                slide_frame_mapping[sid]["face_count_per_frame"][idx] = 0
                slide_frame_mapping[sid]["frames_without_faces"] += 1

    # Overwrite cache with filtered one
    face_crops_cache = filtered_cache

    # Finalize stats
    for sid, m in slide_frame_mapping.items():
        total = int(m["frames_with_faces"]) + int(m["frames_without_faces"])
        m["face_detection_rate"] = (float(m["frames_with_faces"]) / total) if total else 0.0

    t1 = time.time()
    stats = {
        "total_sampled_frames": len(all_frame_indices),
        "processed_frames": total_processed,
        "frames_with_faces_raw": len(all_embeddings_list),
        "frames_with_faces_filtered": filtered_count,
        "face_detection_rate": (filtered_count / len(all_frame_indices)) if all_frame_indices else 0.0,
        "time_sec": t1 - t0,
    }

    return {
        "fps": fps_val,
        "slide_frame_mapping": slide_frame_mapping,
        "face_crops_cache": face_crops_cache,
        "dominant_embedding": dominant_emb.tolist() if dominant_emb is not None else [],
        "stats": stats,
    }
