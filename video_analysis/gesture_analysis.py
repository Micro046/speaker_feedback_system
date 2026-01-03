from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class Gestures:
    """
    YOLOv8 pose-based gesture / body-angle tracker.
    Produces:
      - joint_angles_per_frame: dict[joint_idx] -> list[angle_or_None] aligned to input frames
      - get_result(): (overall_joint_stats, recommendations)
    """

    # Angle triplets: [p1, pmid, p2] => angle at pmid
    # (kept consistent with your earlier mapping)
    joints_to_triplets = {
        8: [8, 6, 5],    # right elbow
        7: [7, 5, 0],    # left elbow
        6: [6, 5, 0],    # right shoulder
        5: [5, 0, 6],    # left shoulder
        10: [10, 8, 6],  # right wrist
        9: [9, 7, 5],    # left wrist
        0: [0, 5, 6],    # head/neck-ish
    }

    joint_names = {
        0: "head",
        5: "left_shoulder",
        6: "right_shoulder",
        7: "left_elbow",
        8: "right_elbow",
        9: "left_wrist",
        10: "right_wrist",
    }

    def __init__(self, model_path: str = "yolov8n-pose.pt", conf: float = 0.3):
        self.model_path = model_path
        self.conf = float(conf)

        logger.info("Loading YOLO pose model from %s", model_path)
        self.model = YOLO(model_path)

        # per-frame angles (aligned to input frames)
        self.joint_angles_per_frame: Dict[int, List[Optional[float]]] = {
            j: [] for j in self.joints_to_triplets.keys()
        }

        # overall movement proxy (avg abs delta angle across frames)
        self._movement_score_by_joint: Dict[int, float] = {}

    @staticmethod
    def _preprocess_frame(frame_bgr: np.ndarray, resize: Tuple[int, int] = (640, 480)) -> np.ndarray:
        frame = cv2.resize(frame_bgr, resize)
        frame = cv2.convertScaleAbs(frame, alpha=1.15, beta=8)  # mild contrast bump
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _angle_deg(p1: np.ndarray, pm: np.ndarray, p2: np.ndarray) -> float:
        v1 = p1 - pm
        v2 = p2 - pm
        n1 = np.linalg.norm(v1) + 1e-6
        n2 = np.linalg.norm(v2) + 1e-6
        cos = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
        return float(np.degrees(np.arccos(cos)))

    @staticmethod
    def _mean_abs_delta(vals: List[Optional[float]]) -> float:
        seq = [v for v in vals if v is not None]
        if len(seq) < 2:
            return 0.0
        diffs = [abs(seq[i] - seq[i - 1]) for i in range(1, len(seq))]
        return float(np.mean(diffs)) if diffs else 0.0

    def process_velocity(self, frames_bgr: List[np.ndarray]) -> Dict[int, float]:
        """
        Runs pose on each frame and fills joint_angles_per_frame.
        Returns per-joint movement score (avg abs delta).
        """
        # reset
        self.joint_angles_per_frame = {j: [] for j in self.joints_to_triplets.keys()}

        for i, frame_bgr in enumerate(frames_bgr):
            try:
                rgb = self._preprocess_frame(frame_bgr)
                preds = self.model.predict(rgb, conf=self.conf, verbose=False)

                # default no detection -> None for all joints
                if not preds or preds[0].keypoints is None or preds[0].keypoints.xy is None:
                    for j in self.joint_angles_per_frame:
                        self.joint_angles_per_frame[j].append(None)
                    continue

                if preds[0].keypoints.xy.shape[0] == 0 or preds[0].keypoints.conf is None:
                    for j in self.joint_angles_per_frame:
                        self.joint_angles_per_frame[j].append(None)
                    continue

                keypoints = preds[0].keypoints.xy[0].cpu().numpy()
                confs = preds[0].keypoints.conf[0].cpu().numpy()

                # landmarks: idx -> (x,y,conf)
                landmarks = {k: (keypoints[k][0], keypoints[k][1], float(confs[k])) for k in range(len(keypoints))}

                for joint, triplet in self.joints_to_triplets.items():
                    if all((t in landmarks) and (landmarks[t][2] >= 0.30) for t in triplet):
                        p1 = np.array(landmarks[triplet[0]][:2], dtype=np.float32)
                        pm = np.array(landmarks[triplet[1]][:2], dtype=np.float32)
                        p2 = np.array(landmarks[triplet[2]][:2], dtype=np.float32)
                        ang = self._angle_deg(p1, pm, p2)
                        self.joint_angles_per_frame[joint].append(float(ang))
                    else:
                        self.joint_angles_per_frame[joint].append(None)

            except Exception as e:
                logger.warning("Pose error on frame %d: %s", i, e)
                for j in self.joint_angles_per_frame:
                    self.joint_angles_per_frame[j].append(None)

        # movement score (proxy for how much you move)
        self._movement_score_by_joint = {
            j: self._mean_abs_delta(vals) for j, vals in self.joint_angles_per_frame.items()
        }
        return dict(self._movement_score_by_joint)

    def get_result(self):
        """
        Returns:
          joint_stats: {joint_name: [movement_score]}  (keeps your earlier style)
          recommendations: {joint_name: "..." }
        """
        joint_stats = {}
        recommendations = {}

        for j, score in self._movement_score_by_joint.items():
            name = self.joint_names.get(j, str(j))
            joint_stats[name] = [round(float(score), 3)]

            # simple, practical heuristics (tune later)
            if score == 0:
                recommendations[name] = "No reliable pose data; ensure upper body is visible and lighting is good."
            elif score > 18:
                recommendations[name] = f"High movement ({score:.1f}). Slow this joint down for a calmer presence."
            elif score < 6:
                recommendations[name] = f"Very low movement ({score:.1f}). Add a bit more expressiveness."
            else:
                recommendations[name] = f"Movement looks balanced ({score:.1f})."

        return joint_stats, recommendations


class GestureConfig:
    """
    Configuration for gesture sampling and processing.
    """

    def __init__(
        self,
        *,
        frames_per_slide_max: int = 6,
        resize_w: int = 640,
        max_total_frames: int = 350,
    ):
        self.frames_per_slide_max = int(frames_per_slide_max)
        self.resize_w = int(resize_w)
        self.max_total_frames = int(max_total_frames)


def _sample_frames_for_slide(
    slide_frame_mapping: Dict[int, Dict[str, Any]],
    frames_per_slide_max: int,
    max_total_frames: int,
) -> List[Dict[str, Any]]:
    picked: List[Dict[str, Any]] = []
    for sid, info in slide_frame_mapping.items():
        indices = list(info.get("frame_indices", []))[: frames_per_slide_max]
        for idx in indices:
            picked.append({
                "slide_id": int(sid),
                "frame_idx": int(idx),
            })
    picked.sort(key=lambda r: (r["slide_id"], r["frame_idx"]))
    return picked[:max_total_frames]


def _read_frames(cap: cv2.VideoCapture, frame_indices: List[int]) -> List[np.ndarray]:
    frames: List[np.ndarray] = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok and frame is not None:
            frames.append(frame)
    return frames


def analyze_gestures_from_video(
    video_path: str,
    slide_frame_mapping: Dict[int, Dict[str, Any]],
    gesture_detector: Gestures,
    idx_to_slide: Optional[Dict[int, Dict[str, Any]]] = None,
    *,
    cfg: Optional[GestureConfig] = None,
) -> Dict[str, Any]:
    """
    Analyze gestures from sampled frames per slide.
    Returns:
      {
        overall: {joint_statistics, recommendations, frames_processed},
        slide_summaries: {slide_id: {joint_statistics, detection_rate, slide_content}},
        issues: {slide_id: [issues...]},
        meta: {frames_used, slides_covered, frames_per_slide_max}
      }
    """
    cfg = cfg or GestureConfig()
    idx_to_slide = idx_to_slide or {}

    picked = _sample_frames_for_slide(
        slide_frame_mapping,
        frames_per_slide_max=cfg.frames_per_slide_max,
        max_total_frames=cfg.max_total_frames,
    )
    if not picked:
        return {
            "overall": {"joint_statistics": {}, "recommendations": {}, "frames_processed": 0},
            "slide_summaries": {},
            "issues": {},
            "meta": {"frames_used": 0, "slides_covered": 0, "frames_per_slide_max": cfg.frames_per_slide_max},
        }

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    # Group frames by slide
    frames_by_slide: Dict[int, List[int]] = {}
    for rec in picked:
        frames_by_slide.setdefault(rec["slide_id"], []).append(rec["frame_idx"])

    # Overall analysis across all frames
    all_indices = [rec["frame_idx"] for rec in picked]
    all_frames = _read_frames(cap, all_indices)
    movement_scores = gesture_detector.process_velocity(all_frames)
    joint_stats, recommendations = gesture_detector.get_result()

    # Per-slide analysis
    slide_summaries: Dict[str, Dict[str, Any]] = {}
    issues: Dict[str, List[str]] = {}

    for sid, idxs in frames_by_slide.items():
        frames = _read_frames(cap, idxs)
        if not frames:
            slide_summaries[str(sid)] = {
                "joint_statistics": {},
                "detection_rate": 0.0,
                "slide_content": idx_to_slide.get(int(sid), {}).get("slide_content", ""),
            }
            issues[str(sid)] = ["No readable frames for gesture analysis."]
            continue

        gesture_detector.process_velocity(frames)
        s_stats, _ = gesture_detector.get_result()
        detection_rate = sum(
            1 for jvals in gesture_detector.joint_angles_per_frame.values() for v in jvals if v is not None
        )
        total_vals = sum(len(v) for v in gesture_detector.joint_angles_per_frame.values())
        det_ratio = float(detection_rate) / float(total_vals) if total_vals else 0.0

        slide_summaries[str(sid)] = {
            "joint_statistics": s_stats,
            "detection_rate": round(det_ratio, 3),
            "slide_content": idx_to_slide.get(int(sid), {}).get("slide_content", ""),
        }

        slide_issues = []
        if det_ratio < 0.4:
            slide_issues.append("Low gesture detection rate; ensure upper body is visible.")
        if slide_issues:
            issues[str(sid)] = slide_issues

    cap.release()

    return {
        "overall": {
            "joint_statistics": joint_stats,
            "recommendations": recommendations,
            "frames_processed": len(all_frames),
        },
        "slide_summaries": slide_summaries,
        "issues": issues,
        "meta": {
            "frames_used": len(picked),
            "slides_covered": len(frames_by_slide),
            "frames_per_slide_max": cfg.frames_per_slide_max,
        },
    }
