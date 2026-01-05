from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class Gestures:
    """
    YOLOv8 pose-based gesture / body-angle tracker.

    Produces:
      - joint_angles_per_frame: dict[joint_idx] -> list[angle_or_None] aligned to input frames
      - frame_pose_ok: list[bool] aligned to input frames (at least K joints available)
      - get_result(): (overall_joint_stats, recommendations)
    """

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

    def __init__(self, model_path: str = "yolov8n-pose.pt", conf: float = 0.3, kp_conf: float = 0.30):
        self.model_path = model_path
        self.conf = float(conf)
        self.kp_conf = float(kp_conf)

        logger.info("Loading YOLO pose model from %s", model_path)
        self.model = YOLO(model_path)

        self.joint_angles_per_frame: Dict[int, List[Optional[float]]] = {j: [] for j in self.joints_to_triplets.keys()}
        self.frame_pose_ok: List[bool] = []
        self._movement_score_by_joint: Dict[int, float] = {}

    @staticmethod
    def _preprocess_frame(frame_bgr: np.ndarray, resize: Tuple[int, int]) -> np.ndarray:
        frame = cv2.resize(frame_bgr, resize)
        frame = cv2.convertScaleAbs(frame, alpha=1.15, beta=8)
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

    def process_velocity(
        self,
        frames_bgr: List[np.ndarray],
        *,
        resize_wh: Tuple[int, int] = (640, 480),
        min_valid_joints_per_frame: int = 3,
    ) -> Dict[int, float]:
        """
        Runs pose on each frame and fills joint_angles_per_frame.
        Also sets frame_pose_ok for coverage.
        """
        self.joint_angles_per_frame = {j: [] for j in self.joints_to_triplets.keys()}
        self.frame_pose_ok = []

        for i, frame_bgr in enumerate(frames_bgr):
            try:
                rgb = self._preprocess_frame(frame_bgr, resize_wh)
                preds = self.model.predict(rgb, conf=self.conf, verbose=False)

                if not preds or preds[0].keypoints is None or preds[0].keypoints.xy is None:
                    for j in self.joint_angles_per_frame:
                        self.joint_angles_per_frame[j].append(None)
                    self.frame_pose_ok.append(False)
                    continue

                if preds[0].keypoints.xy.shape[0] == 0 or preds[0].keypoints.conf is None:
                    for j in self.joint_angles_per_frame:
                        self.joint_angles_per_frame[j].append(None)
                    self.frame_pose_ok.append(False)
                    continue

                keypoints = preds[0].keypoints.xy[0].cpu().numpy()
                confs = preds[0].keypoints.conf[0].cpu().numpy()

                landmarks = {
                    k: (float(keypoints[k][0]), float(keypoints[k][1]), float(confs[k]))
                    for k in range(len(keypoints))
                }

                valid_joint_count = 0
                for joint, triplet in self.joints_to_triplets.items():
                    if all((t in landmarks) and (landmarks[t][2] >= self.kp_conf) for t in triplet):
                        p1 = np.array(landmarks[triplet[0]][:2], dtype=np.float32)
                        pm = np.array(landmarks[triplet[1]][:2], dtype=np.float32)
                        p2 = np.array(landmarks[triplet[2]][:2], dtype=np.float32)
                        ang = self._angle_deg(p1, pm, p2)
                        self.joint_angles_per_frame[joint].append(float(ang))
                        valid_joint_count += 1
                    else:
                        self.joint_angles_per_frame[joint].append(None)

                self.frame_pose_ok.append(valid_joint_count >= min_valid_joints_per_frame)

            except Exception as e:
                logger.warning("Pose error on frame %d: %s", i, e)
                for j in self.joint_angles_per_frame:
                    self.joint_angles_per_frame[j].append(None)
                self.frame_pose_ok.append(False)

        self._movement_score_by_joint = {j: self._mean_abs_delta(vals) for j, vals in self.joint_angles_per_frame.items()}
        return dict(self._movement_score_by_joint)

    def get_result(self):
        joint_stats = {}
        recommendations = {}

        for j, score in self._movement_score_by_joint.items():
            name = self.joint_names.get(j, str(j))
            joint_stats[name] = [round(float(score), 3)]

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
    """
    Uniform-ish sampling: take evenly spaced indices from slide_frame_mapping frame_indices.
    (They are already uniform from your face-cache sampler.)
    """
    picked: List[Dict[str, Any]] = []
    for sid, info in slide_frame_mapping.items():
        indices = list(info.get("frame_indices", []))
        if not indices:
            continue
        # take up to frames_per_slide_max spread across indices
        if len(indices) <= frames_per_slide_max:
            chosen = indices
        else:
            take = frames_per_slide_max
            pos = np.linspace(0, len(indices) - 1, take)
            chosen = [indices[int(round(p))] for p in pos]
        for idx in chosen:
            picked.append({"slide_id": int(sid), "frame_idx": int(idx)})

    picked.sort(key=lambda r: (r["slide_id"], r["frame_idx"]))
    return picked[:max_total_frames]


def _read_frames_with_meta(cap: cv2.VideoCapture, frame_indices: List[int]) -> List[Tuple[int, np.ndarray]]:
    out: List[Tuple[int, np.ndarray]] = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok and frame is not None:
            out.append((int(idx), frame))
    return out


def analyze_gestures_from_video(
    video_path: str,
    slide_frame_mapping: Dict[int, Dict[str, Any]],
    gesture_detector: Gestures,
    idx_to_slide: Optional[Dict[int, Dict[str, Any]]] = None,
    *,
    cfg: Optional[GestureConfig] = None,
) -> Dict[str, Any]:
    cfg = cfg or GestureConfig()
    idx_to_slide = idx_to_slide or {}

    picked = _sample_frames_for_slide(
        slide_frame_mapping,
        frames_per_slide_max=cfg.frames_per_slide_max,
        max_total_frames=cfg.max_total_frames,
    )
    if not picked:
        return {
            "overall": {"joint_statistics": {}, "recommendations": {}, "frames_processed": 0, "pose_coverage": 0.0},
            "slide_summaries": {},
            "issues": {},
            "meta": {"frames_used": 0, "slides_covered": 0, "frames_per_slide_max": cfg.frames_per_slide_max},
        }

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    resize_h = int(round(cfg.resize_w * 0.75))
    resize_wh = (cfg.resize_w, resize_h)

    # Group frames by slide
    frames_by_slide: Dict[int, List[int]] = defaultdict(list)
    for rec in picked:
        frames_by_slide[int(rec["slide_id"])].append(int(rec["frame_idx"]))

    # ---- Overall analysis across all frames ----
    all_indices = [int(rec["frame_idx"]) for rec in picked]
    all_pairs = _read_frames_with_meta(cap, all_indices)
    all_frames = [f for _, f in all_pairs]

    gesture_detector.process_velocity(all_frames, resize_wh=resize_wh)
    joint_stats, recommendations = gesture_detector.get_result()

    overall_pose_cov = float(sum(gesture_detector.frame_pose_ok) / max(len(gesture_detector.frame_pose_ok), 1))

    # ---- Per-slide analysis ----
    slide_summaries: Dict[str, Dict[str, Any]] = {}
    issues: Dict[str, List[str]] = {}

    for sid, idxs in frames_by_slide.items():
        pairs = _read_frames_with_meta(cap, idxs)
        frames = [f for _, f in pairs]
        if not frames:
            slide_summaries[str(sid)] = {
                "joint_statistics": {},
                "pose_coverage": 0.0,
                "frames_sampled": int(len(idxs)),
                "frames_processed": 0,
                "evidence_frames": [],
                "slide_content": idx_to_slide.get(int(sid), {}).get("slide_content", ""),
                "audio_content": idx_to_slide.get(int(sid), {}).get("audio_content", ""),
            }
            issues[str(sid)] = ["No readable frames for gesture analysis."]
            continue

        gesture_detector.process_velocity(frames, resize_wh=resize_wh)
        s_stats, s_recs = gesture_detector.get_result()
        pose_cov = float(sum(gesture_detector.frame_pose_ok) / max(len(gesture_detector.frame_pose_ok), 1))

        # evidence frames: choose frames where pose_ok=True (up to 3)
        evidence_frames = []
        for (frame_idx, _), ok in zip(pairs, gesture_detector.frame_pose_ok):
            if ok:
                evidence_frames.append({
                    "frame_idx": int(frame_idx),
                    "t_sec": float(frame_idx) / float(fps),
                })
        evidence_frames = evidence_frames[:3]

        slide_summaries[str(sid)] = {
            "joint_statistics": s_stats,
            "recommendations": s_recs,
            "pose_coverage": round(pose_cov, 3),
            "frames_sampled": int(len(idxs)),
            "frames_processed": int(len(frames)),
            "evidence_frames": evidence_frames,
            "slide_content": idx_to_slide.get(int(sid), {}).get("slide_content", ""),
            "audio_content": idx_to_slide.get(int(sid), {}).get("audio_content", ""),
        }

        slide_issues = []
        if pose_cov < 0.35:
            slide_issues.append("Low pose coverage; ensure upper body/hands are visible.")
        if slide_issues:
            issues[str(sid)] = slide_issues

    cap.release()

    return {
        "overall": {
            "joint_statistics": joint_stats,
            "recommendations": recommendations,
            "frames_processed": len(all_frames),
            "pose_coverage": round(overall_pose_cov, 3),
        },
        "slide_summaries": slide_summaries,
        "issues": issues,
        "meta": {
            "frames_used": len(picked),
            "slides_covered": len(frames_by_slide),
            "frames_per_slide_max": cfg.frames_per_slide_max,
            "fps": float(fps),
            "resize_wh": {"w": int(cfg.resize_w), "h": int(resize_h)},
        },
    }
