from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict, Counter

import cv2
import numpy as np
import torch


def _cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@dataclass
class GazeHeuristics:
    # Angles in degrees
    pitch_down_thresh: float = 12.0
    yaw_side_thresh: float = 18.0
    pitch_center_thresh: float = 10.0
    yaw_center_thresh: float = 7.0
    # Positive values reduce downward bias if the camera is above eye level.
    pitch_offset: float = 0.0


class MediaPipeGazeDirection:
    """
    Head-pose-based gaze estimator using PnP.
    Returns semantic labels: audience, script_notes, screen_left, screen_right.
    """
    def __init__(
        self,
        *,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        heuristics: Optional[GazeHeuristics] = None,
    ):
        try:
            import mediapipe as mp
        except Exception as e:
            raise RuntimeError(
                "mediapipe is required for MediaPipeGazeDirection."
            ) from e

        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            refine_landmarks=True,
            max_num_faces=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._cfg = heuristics or GazeHeuristics()

        # 3D Face Model Points (Generic)
        self.face_3d = np.array(
            [
                (0.0, 0.0, 0.0),             # Nose tip
                (0.0, -330.0, -65.0),        # Chin
                (-225.0, 170.0, -135.0),     # Left eye left corner
                (225.0, 170.0, -135.0),      # Right eye right corner
                (-150.0, -150.0, -125.0),    # Left mouth corner
                (150.0, -150.0, -125.0)      # Right mouth corner
            ],
            dtype=np.float64,
        )

        # Corresponding MediaPipe indices
        self.face_2d_idx = [1, 152, 33, 263, 61, 291]

    def process_frame(self, face_crop_bgr) -> Tuple[str, float]:
        """
        Returns (semantic_label, confidence_score)
        Labels: "audience", "script_notes", "screen_left", "screen_right"
        """
        h, w, _ = face_crop_bgr.shape
        rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
        results = self._mesh.process(rgb)

        if not results.multi_face_landmarks:
            return "no_face", 0.0

        landmarks = results.multi_face_landmarks[0].landmark
        face_2d = []
        for idx in self.face_2d_idx:
            lm = landmarks[idx]
            x, y = int(lm.x * w), int(lm.y * h)
            face_2d.append([x, y])

        face_2d = np.array(face_2d, dtype=np.float64)

        focal_length = 1.0 * w
        cam_matrix = np.array(
            [
                [focal_length, 0, w / 2],
                [0, focal_length, h / 2],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )

        dist_matrix = np.zeros((4, 1), dtype=np.float64)

        success, rot_vec, _ = cv2.solvePnP(
            self.face_3d, face_2d, cam_matrix, dist_matrix
        )

        if not success:
            return "unknown", 0.0

        rmat, _ = cv2.Rodrigues(rot_vec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

        # Angles: x=Pitch, y=Yaw, z=Roll
        pitch = float(angles[0] * 360)
        yaw = float(angles[1] * 360)
        pitch_adj = pitch - float(self._cfg.pitch_offset or 0.0)

        # Semantic Mapping
        if pitch_adj > self._cfg.pitch_down_thresh:
            label = "script_notes"
            conf = min(1.0, (pitch_adj - self._cfg.pitch_down_thresh) / 15.0)
        elif abs(yaw) > self._cfg.yaw_side_thresh:
            label = "screen_right" if yaw > 0 else "screen_left"
            conf = min(1.0, (abs(yaw) - self._cfg.yaw_side_thresh) / 20.0)
        else:
            label = "audience"
            conf = 1.0 - max(
                abs(yaw) / max(self._cfg.yaw_center_thresh, 1.0),
                abs(pitch_adj) / max(self._cfg.pitch_center_thresh, 1.0),
            )
            conf = float(max(0.4, min(1.0, conf)))

        return label, float(conf)

    @staticmethod
    def summarize_gaze(labels) -> Dict[str, object]:
        valid = [l for l in labels if l not in {"no_face", "unknown"}]
        if not valid:
            return {
                "dominant_focus": "unknown",
                "focus_dist": {"audience": 0.0, "script": 0.0, "slides": 0.0},
                "recommendation": "Face not visible enough to determine gaze.",
            }

        counts = Counter(valid)
        bs = len(valid)

        audience_pct = (counts["audience"] / bs) * 100
        script_pct = (counts["script_notes"] / bs) * 100
        slide_pct = ((counts["screen_left"] + counts["screen_right"]) / bs) * 100

        dist = {
            "audience": round(audience_pct, 1),
            "script": round(script_pct, 1),
            "slides": round(slide_pct, 1),
        }

        if script_pct > 30:
            rec = "Detected frequent downward gaze. Try to rely less on notes and engage with the audience."
        elif slide_pct > 40:
            rec = "You are looking at the screen/slides often. Face the audience while explaining."
        elif audience_pct > 70:
            rec = "Excellent eye contact with the audience/camera."
        else:
            rec = "Balanced gaze, but room for improvement in maintaining eye contact."

        return {
            "dominant_focus": max(dist, key=dist.get),
            "focus_dist": dist,
            "recommendation": rec,
            "raw_distribution": dict(counts),
        }

    def close(self):
        if self._mesh:
            self._mesh.close()


@dataclass
class GazeConfig:
    frames_per_slide_max: int = 6
    min_face_conf: float = 0.55
    min_gaze_conf: float = 0.2
    expand_scale: float = 1.3
    min_face_size: int = 48
    max_total_frames: int = 300
    top_evidence_frames: int = 3  # store best frames per slide for NAT citations
    min_valid_frames_per_slide: int = 2
    min_coverage_ratio: float = 0.2
    min_overall_valid_ratio: float = 0.3


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
    valid_labels: List[str] = []
    total_samples = 0

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
        total_samples += 1

        slide_records[sid].append({
            "frame_idx": int(rec["frame_idx"]),
            "t_sec": float(rec["frame_idx"]) / float(fps),
            "gaze": gaze,
            "gaze_conf": conf,
            "face_conf": float(rec["face_conf"]),
        })

        if conf >= cfg.min_gaze_conf and gaze not in {"no_face", "unknown"}:
            valid_labels.append(gaze)

    cap.release()
    _cleanup()

    # ---- overall summary ----
    if valid_labels:
        overall_summ = gaze_estimator.summarize_gaze(valid_labels)
    else:
        overall_summ = {
            "dominant_focus": "unknown",
            "focus_dist": {"audience": 0.0, "script": 0.0, "slides": 0.0},
            "recommendation": "Insufficient face visibility to assess gaze.",
        }

    valid_ratio = (len(valid_labels) / max(total_samples, 1)) if total_samples else 0.0
    overall = {
        "dominant_focus": overall_summ.get("dominant_focus"),
        "focus_dist": overall_summ.get("focus_dist", {}),
        "focus_distribution": overall_summ.get("focus_dist", {}),
        "recommendation": overall_summ.get("recommendation", ""),
        "valid_gaze_ratio": round(float(valid_ratio), 3),
        "valid_frames": int(len(valid_labels)),
        "sampled_frames": int(total_samples),
        "data_quality": "low" if valid_ratio < cfg.min_overall_valid_ratio else "ok",
    }

    slide_summaries: Dict[str, Any] = {}
    issues: Dict[str, Any] = {}

    for sid in sorted(slide_frame_mapping.keys()):
        sampled = len(slide_frame_mapping[sid].get("frame_indices", []))
        recs = slide_records.get(sid, [])

        valid = [r for r in recs if r["gaze"] not in {"no_face", "unknown"} and r["gaze_conf"] >= cfg.min_gaze_conf]

        labels = [r["gaze"] for r in valid]
        if labels:
            summ = gaze_estimator.summarize_gaze(labels)
        else:
            summ = {
                "dominant_focus": "unknown",
                "focus_dist": {"audience": 0.0, "script": 0.0, "slides": 0.0},
                "recommendation": "Insufficient face visibility to assess gaze.",
            }

        evidence = sorted(valid, key=lambda r: r["gaze_conf"], reverse=True)[: cfg.top_evidence_frames]

        coverage_ratio = float(round(len(valid) / max(sampled, 1), 3))

        counts = Counter(labels)
        dist_lr = {
            "left": round((counts.get("screen_left", 0) / max(len(labels), 1)) * 100, 1),
            "right": round((counts.get("screen_right", 0) / max(len(labels), 1)) * 100, 1),
            "center": round((counts.get("audience", 0) / max(len(labels), 1)) * 100, 1),
        }

        slide_summaries[str(sid)] = {
            "dominant_focus": summ["dominant_focus"],
            "focus_distribution": summ["focus_dist"], # {audience, script, slides}
            "focus_dist": summ["focus_dist"],
            "distribution": dist_lr,  # left/right/center for downstream aggregation
            "recommendation": summ["recommendation"],
            "sampled_frames": int(sampled),
            "valid_frames": int(len(valid)),
            "coverage_ratio": coverage_ratio,
            "evidence_frames": evidence,
            "slide_content": idx_to_slide.get(int(sid), {}).get("slide_content", ""),
            "data_quality": {
                "min_valid_frames": cfg.min_valid_frames_per_slide,
                "min_coverage_ratio": cfg.min_coverage_ratio,
                "is_reliable": bool(
                    len(valid) >= cfg.min_valid_frames_per_slide and coverage_ratio >= cfg.min_coverage_ratio
                ),
            },
        }

        # Issues based on semantic labels
        slide_issues = []
        if coverage_ratio < cfg.min_coverage_ratio or len(valid) < cfg.min_valid_frames_per_slide:
            slide_issues.append("Low face visibility for gaze analysis")
        else:
            if summ["focus_dist"]["script"] > 30.0:
                slide_issues.append("Reading from script")
            if summ["focus_dist"]["slides"] > 40.0:
                slide_issues.append("Staring at slides")
            
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
                "min_gaze_conf": cfg.min_gaze_conf,
                "expand_scale": cfg.expand_scale,
                "min_face_size": cfg.min_face_size,
                "min_valid_frames_per_slide": cfg.min_valid_frames_per_slide,
                "min_coverage_ratio": cfg.min_coverage_ratio,
                "min_overall_valid_ratio": cfg.min_overall_valid_ratio,
            }
        },
    }
