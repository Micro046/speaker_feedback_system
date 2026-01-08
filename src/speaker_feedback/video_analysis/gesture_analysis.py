from __future__ import annotations

import gc
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

import cv2
import numpy as np
import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)


@dataclass
class GestureHeuristics:
    hand_to_face_ratio: float = 0.6
    arms_crossed_wrist_ratio: float = 0.5
    arms_crossed_chest_y_ratio: float = 0.3
    open_palms_ratio: float = 1.2


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

    def __init__(
        self,
        model_path: str = "yolov8n-pose.pt",
        conf: float = 0.3,
        kp_conf: float = 0.30,
        heuristics: Optional[GestureHeuristics] = None,
    ):
        self.model_path = model_path
        self.conf = float(conf)
        self.kp_conf = float(kp_conf)
        self.heuristics = heuristics or GestureHeuristics()

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

    def process_semantic_gestures(
        self,
        frames_bgr: List[np.ndarray],
        *,
        resize_wh: Tuple[int, int] = (640, 480),
    ) -> Dict[str, Any]:
        """
        Runs pose detection and computes semantic body language states.
        """
        self.frame_pose_ok = []
        
        # State counters
        states = {
            "arms_crossed": 0,
            "hands_in_pockets": 0,
            "hand_to_face": 0,
            "hands_behind_back": 0,
            "open_palms": 0,
            "total_valid": 0
        }
        self.frame_states_per_frame = [] # List of list of strings
        
        for frame_bgr in frames_bgr:
            try:
                rgb = self._preprocess_frame(frame_bgr, resize_wh)
                preds = self.model.predict(rgb, conf=self.conf, verbose=False)
                
                if not preds or preds[0].keypoints is None or preds[0].keypoints.xy is None:
                    self.frame_pose_ok.append(False)
                    continue

                kps_all = preds[0].keypoints.xy
                confs_all = preds[0].keypoints.conf
                boxes = preds[0].boxes.xyxy if preds[0].boxes is not None else None

                # choose the best person (largest box, then avg keypoint confidence)
                best_idx = 0
                best_area = -1.0
                best_conf = -1.0
                for i in range(kps_all.shape[0]):
                    avg_conf = float(confs_all[i].mean().item()) if confs_all is not None else 0.0
                    area = 0.0
                    if boxes is not None and i < boxes.shape[0]:
                        x1, y1, x2, y2 = boxes[i].cpu().numpy().tolist()
                        area = max(0.0, (x2 - x1) * (y2 - y1))
                    if area > best_area or (area == best_area and avg_conf > best_conf):
                        best_area = area
                        best_conf = avg_conf
                        best_idx = i

                kps = kps_all[best_idx].cpu().numpy()
                confs = confs_all[best_idx].cpu().numpy()
                
                # Indices: 0:Nose, 5:LShoulder, 6:RShoulder, 7:LElbow, 8:RElbow, 9:LWrist, 10:RWrist, 11:LHip, 12:RHip
                # Check visibility
                needed = [0, 5, 6, 9, 10] # Min needed for most
                if any(confs[i] < self.kp_conf for i in needed):
                    self.frame_pose_ok.append(False)
                    continue
                
                self.frame_pose_ok.append(True)
                states["total_valid"] += 1
                
                # Coords
                nose = kps[0]
                ls, rs = kps[5], kps[6]
                lw, rw = kps[9], kps[10]
                lh, rh = kps[11], kps[12] if len(kps) > 12 else (None, None)
                
                # 1. Hand to Face (Wrists near Nose)
                # Normalize dist by shoulder width
                shoulder_width = np.linalg.norm(ls - rs) + 1e-6
                dist_lw_nose = np.linalg.norm(lw - nose) / shoulder_width
                dist_rw_nose = np.linalg.norm(rw - nose) / shoulder_width
                
                if dist_lw_nose < self.heuristics.hand_to_face_ratio or dist_rw_nose < self.heuristics.hand_to_face_ratio:
                    states["hand_to_face"] += 1
                    # TODO: Add timestamp to events if needed (passed in meta?)
                    
                # 2. Arms Crossed (Wrists close to opposite elbows/shoulders or close to each other near chest)
                # Simplification: wrists close together + wrists between shoulders vertically
                chest_y = (ls[1] + rs[1]) / 2.0 + (shoulder_width * self.heuristics.arms_crossed_chest_y_ratio)
                if lw[1] > ls[1] and rw[1] > rs[1]: # Below shoulders
                    wrist_dist = np.linalg.norm(lw - rw) / shoulder_width
                    if wrist_dist < self.heuristics.arms_crossed_wrist_ratio and abs(lw[1] - chest_y) < shoulder_width * 0.5:
                        states["arms_crossed"] += 1
                        
                # 3. Hands in Pockets / Hidden (Wrists below hips)
                if lh is not None and rh is not None:
                     # Check conf for hips
                     if confs[11] > self.kp_conf and confs[12] > self.kp_conf:
                        if lw[1] > lh[1] and rw[1] > rh[1]:
                             states["hands_in_pockets"] += 1
                             
                # 4. Open Palms (Wrists wide apart + Elbows out)
                wrist_spread = np.linalg.norm(lw - rw) / shoulder_width
                if wrist_spread > self.heuristics.open_palms_ratio:
                    states["open_palms"] += 1
                
                # Store per-frame state for event generation (handled by caller currently or we can return it)
                # Let's return a list of explicit states for this frame
                frame_states = []
                if dist_lw_nose < self.heuristics.hand_to_face_ratio or dist_rw_nose < self.heuristics.hand_to_face_ratio:
                    frame_states.append("Hand to Face")
                if lw[1] > ls[1] and rw[1] > rs[1]:
                    wrist_dist = np.linalg.norm(lw - rw) / shoulder_width
                    if wrist_dist < self.heuristics.arms_crossed_wrist_ratio and abs(lw[1] - chest_y) < shoulder_width * 0.5:
                        frame_states.append("Arms Crossed")
                if lh is not None and rh is not None and confs[11] > self.kp_conf and confs[12] > self.kp_conf:
                    if lw[1] > lh[1] and rw[1] > rh[1]:
                        frame_states.append("Hands in Pockets")
                if wrist_spread > self.heuristics.open_palms_ratio:
                    frame_states.append("Open Palms")
                
                self.frame_states_per_frame.append(frame_states)

            except Exception as e:
                self.frame_pose_ok.append(False)
                self.frame_states_per_frame.append([])
        
        # Calculate percentages
        total = max(states["total_valid"], 1)
        return {
            "arms_crossed_pct": round((states["arms_crossed"] / total) * 100, 1),
            "hands_in_pockets_pct": round((states["hands_in_pockets"] / total) * 100, 1),
            "hand_to_face_pct": round((states["hand_to_face"] / total) * 100, 1),
            "open_palms_pct": round((states["open_palms"] / total) * 100, 1),
            "pose_coverage_pct": round((states["total_valid"] / max(len(frames_bgr), 1)) * 100, 1),
            "frame_states": self.frame_states_per_frame
        }

    def get_result(self, stats: Dict[str, Any]):
        recs = []
        
        if stats["pose_coverage_pct"] < 20:
            recs.append("Ensure your upper body and hands are visible.")
        
        if stats["arms_crossed_pct"] > 30:
            recs.append("Frequent arm crossing detected. Try to keep an open posture.")
            
        if stats["hand_to_face_pct"] > 20:
             recs.append("Frequent face touching. This can signal nervousness.")
             
        if stats["hands_in_pockets_pct"] > 40:
             recs.append("Hands often low/hidden. Use gestures to emphasize points.")
             
        if stats["open_palms_pct"] > 30:
             recs.append("Good use of open, expansive gestures.")
             
        if not recs:
            recs.append("Body language looks neutral.")
            
        return stats, recs


class GestureConfig:
    def __init__(
        self,
        *,
        frames_per_slide_max: int = 6,
        resize_w: int = 640,
        max_total_frames: int = 350,
        min_event_frames: int = 3,
        min_pose_coverage_ratio: float = 0.35,
        top_evidence_frames: int = 3,
    ):
        self.frames_per_slide_max = int(frames_per_slide_max)
        self.resize_w = int(resize_w)
        self.max_total_frames = int(max_total_frames)
        self.min_event_frames = int(min_event_frames)
        self.min_pose_coverage_ratio = float(min_pose_coverage_ratio)
        self.top_evidence_frames = int(top_evidence_frames)


def _event_intervals(
    frame_states: List[List[str]],
    frame_times: List[float],
    min_frames: int,
) -> List[Dict[str, Any]]:
    events = []
    if not frame_states or not frame_times:
        return events

    states = {"Hand to Face", "Arms Crossed", "Hands in Pockets", "Open Palms"}
    for state in states:
        i = 0
        while i < len(frame_states):
            if state in frame_states[i]:
                start = i
                i += 1
                while i < len(frame_states) and state in frame_states[i]:
                    i += 1
                length = i - start
                if length >= min_frames:
                    events.append({
                        "event": state,
                        "start_time": round(float(frame_times[start]), 2),
                        "end_time": round(float(frame_times[i - 1]), 2),
                        "frames": int(length),
                    })
            else:
                i += 1

    events.sort(key=lambda e: e["start_time"])
    return events


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
    # ---- Overall analysis across all frames ----
    all_indices = [int(rec["frame_idx"]) for rec in picked]
    all_pairs = _read_frames_with_meta(cap, all_indices)
    all_frames = [f for _, f in all_pairs]

    overall_stats_raw = gesture_detector.process_semantic_gestures(all_frames, resize_wh=resize_wh)
    joint_stats, recommendations = gesture_detector.get_result(overall_stats_raw)

    overall_pose_cov = float(overall_stats_raw["pose_coverage_pct"] / 100.0)

    # ---- Per-slide analysis ----
    slide_summaries: Dict[str, Dict[str, Any]] = {}
    issues: Dict[str, List[str]] = {}

    for sid, idxs in frames_by_slide.items():
        pairs = _read_frames_with_meta(cap, idxs)
        frames = [f for _, f in pairs]
        frame_times = [float(frame_idx) / float(fps) for frame_idx, _ in pairs]
        if not frames:
            slide_summaries[str(sid)] = {
                "joint_statistics": {},
                "pose_coverage": 0.0,
                "frames_sampled": int(len(idxs)),
                "frames_processed": 0,
                "evidence_frames": [],
                "events": [],
                "slide_content": idx_to_slide.get(int(sid), {}).get("slide_content", ""),
                "audio_content": idx_to_slide.get(int(sid), {}).get("audio_content", ""),
            }
            issues[str(sid)] = ["No readable frames for gesture analysis."]
            continue

        s_stats = gesture_detector.process_semantic_gestures(frames, resize_wh=resize_wh)
        _, s_recs = gesture_detector.get_result(s_stats)
        pose_cov = float(s_stats["pose_coverage_pct"] / 100.0)

        # evidence frames: choose frames where pose_ok=True
        evidence_frames = []
        
        for i, ((frame_idx, _), ok) in enumerate(zip(pairs, gesture_detector.frame_pose_ok)):
            t_sec = float(frame_idx) / float(fps)
            if ok:
                evidence_frames.append({
                    "frame_idx": int(frame_idx),
                    "t_sec": t_sec,
                })

        evidence_frames = evidence_frames[: cfg.top_evidence_frames]
        events_list = _event_intervals(
            s_stats.get("frame_states", []),
            frame_times,
            cfg.min_event_frames,
        )

        slide_summaries[str(sid)] = {
            "joint_statistics": s_stats,
            "recommendations": s_recs,
            "pose_coverage": round(pose_cov, 3),
            "frames_sampled": int(len(idxs)),
            "frames_processed": int(len(frames)),
            "evidence_frames": evidence_frames,
            "events": events_list,
            "slide_content": idx_to_slide.get(int(sid), {}).get("slide_content", ""),
            "audio_content": idx_to_slide.get(int(sid), {}).get("audio_content", ""),
            "data_quality": {
                "min_pose_coverage_ratio": cfg.min_pose_coverage_ratio,
                "is_reliable": bool(pose_cov >= cfg.min_pose_coverage_ratio),
            },
        }

        slide_issues = []
        if pose_cov < cfg.min_pose_coverage_ratio:
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
            "data_quality": {
                "min_pose_coverage_ratio": cfg.min_pose_coverage_ratio,
                "is_reliable": bool(overall_pose_cov >= cfg.min_pose_coverage_ratio),
            },
        },
        "slide_summaries": slide_summaries,
        "issues": issues,
        "meta": {
            "frames_used": len(picked),
            "slides_covered": len(frames_by_slide),
            "frames_per_slide_max": cfg.frames_per_slide_max,
            "fps": float(fps),
            "resize_wh": {"w": int(cfg.resize_w), "h": int(resize_h)},
            "min_event_frames": cfg.min_event_frames,
            "min_pose_coverage_ratio": cfg.min_pose_coverage_ratio,
            "top_evidence_frames": cfg.top_evidence_frames,
        },
    }


def gestures_analysis_tool(
    video_path: str,
    slide_frame_mapping: Dict[int, Dict[str, Any]],
    gesture_detector,
    idx_to_slide: Optional[Dict[int, Dict[str, Any]]] = None,
    *,
    frames_per_slide_max: int = 6,
    resize_w: int = 640,
    max_total_frames: int = 350,
    min_event_frames: int = 3,
    min_pose_coverage_ratio: float = 0.35,
    top_evidence_frames: int = 3,
) -> Dict[str, Any]:
    """
    Tool wrapper for gesture / body-language analysis.
    """
    cfg = GestureConfig(
        frames_per_slide_max=frames_per_slide_max,
        resize_w=resize_w,
        max_total_frames=max_total_frames,
        min_event_frames=min_event_frames,
        min_pose_coverage_ratio=min_pose_coverage_ratio,
        top_evidence_frames=top_evidence_frames,
    )

    out = analyze_gestures_from_video(
        video_path=video_path,
        slide_frame_mapping=slide_frame_mapping,
        gesture_detector=gesture_detector,
        idx_to_slide=idx_to_slide,
        cfg=cfg,
    )

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return out


def gesture_analysis_tool(
    video_path: str,
    slide_frame_mapping: Dict[int, Dict[str, Any]],
    gesture_detector,
    idx_to_slide: Optional[Dict[int, Dict[str, Any]]] = None,
    *,
    frames_per_slide_max: int = 6,
    resize_w: int = 640,
    max_total_frames: int = 350,
    min_event_frames: int = 3,
    min_pose_coverage_ratio: float = 0.35,
    top_evidence_frames: int = 3,
) -> Dict[str, Any]:
    """
    Backwards-compatible alias for notebooks or older imports.
    """
    return gestures_analysis_tool(
        video_path=video_path,
        slide_frame_mapping=slide_frame_mapping,
        gesture_detector=gesture_detector,
        idx_to_slide=idx_to_slide,
        frames_per_slide_max=frames_per_slide_max,
        resize_w=resize_w,
        max_total_frames=max_total_frames,
        min_event_frames=min_event_frames,
        min_pose_coverage_ratio=min_pose_coverage_ratio,
        top_evidence_frames=top_evidence_frames,
    )
