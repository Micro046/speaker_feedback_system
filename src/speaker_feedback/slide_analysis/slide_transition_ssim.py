from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import cv2
import numpy as np
import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor


@dataclass
class SlideBBox:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float


@dataclass
class SlideTransitionConfig:
    model_path: Optional[str] = None
    config_path: Optional[str] = None
    sample_every_sec: float = 1.0
    downscale_max_side: int = 960
    score_thresh: float = 0.95
    ssim_thresh: float = 0.85
    debounce_sec: float = 1.0
    min_segment_sec: float = 2.0
    use_full_frame_if_no_predictor: bool = False


class SlideTransitionDetector:
    def __init__(self, predictor=None, config: Optional[SlideTransitionConfig] = None):
        self.predictor = predictor
        self.cfg = config or SlideTransitionConfig()

    def _get_predictor(self):
        if self.predictor is not None:
            return self.predictor
        if self.cfg.model_path and self.cfg.config_path:
            self.predictor = build_predictor(
                self.cfg.model_path,
                self.cfg.config_path,
                score_thresh=self.cfg.score_thresh,
            )
        return self.predictor

    def detect_transitions(self, video_path: str) -> List[Dict[str, Any]]:
        res = detect_transitions_and_segments(
            video_path=video_path,
            model_path=self.cfg.model_path,
            config_path=self.cfg.config_path,
            sample_every_sec=self.cfg.sample_every_sec,
            downscale_max_side=self.cfg.downscale_max_side,
            score_thresh=self.cfg.score_thresh,
            ssim_thresh=self.cfg.ssim_thresh,
            debounce_sec=self.cfg.debounce_sec,
            min_segment_sec=self.cfg.min_segment_sec,
            predictor=self._get_predictor(),
            use_full_frame_if_no_predictor=self.cfg.use_full_frame_if_no_predictor,
        )
        return res["segments"]


def build_predictor(model_path: str, config_path: str, score_thresh: float = 0.95):
    model_path = str(Path(model_path).expanduser().resolve())
    config_path = str(Path(config_path).expanduser().resolve())

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model weights not found: {model_path}")
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    cfg = get_cfg()
    cfg.merge_from_file(config_path)
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float(score_thresh)
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.freeze()
    return DefaultPredictor(cfg)


def _best_bbox_xyxy(frame_bgr: np.ndarray, predictor) -> Optional[Tuple[int, int, int, int, float]]:
    outputs = predictor(frame_bgr)
    inst = outputs["instances"].to("cpu")
    if len(inst) == 0:
        return None

    scores = inst.scores.numpy()
    boxes = inst.pred_boxes.tensor.numpy()
    best = int(scores.argmax())
    x1, y1, x2, y2 = boxes[best].astype(int).tolist()
    return x1, y1, x2, y2, float(scores[best])


def detect_bbox_on_frame(frame_bgr: np.ndarray, predictor, downscale_max_side: int = 960) -> Optional[SlideBBox]:
    """
    Runs Detectron2 on a downscaled copy for speed, then maps bbox back to original frame coords.
    """
    h0, w0 = frame_bgr.shape[:2]

    scale = 1.0
    if max(h0, w0) > downscale_max_side:
        scale = downscale_max_side / max(h0, w0)
        frame_small = cv2.resize(
            frame_bgr, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_AREA
        )
    else:
        frame_small = frame_bgr

    best = _best_bbox_xyxy(frame_small, predictor)
    if best is None:
        return None

    x1s, y1s, x2s, y2s, score = best

    if scale != 1.0:
        inv = 1.0 / scale
        x1 = int(round(x1s * inv))
        y1 = int(round(y1s * inv))
        x2 = int(round(x2s * inv))
        y2 = int(round(y2s * inv))
    else:
        x1, y1, x2, y2 = x1s, y1s, x2s, y2s

    # clamp
    x1 = max(0, min(w0 - 1, x1))
    y1 = max(0, min(h0 - 1, y1))
    x2 = max(1, min(w0, x2))
    y2 = max(1, min(h0, y2))

    return SlideBBox(x1=x1, y1=y1, x2=x2, y2=y2, score=score)


def crop_slide(frame_bgr: np.ndarray, bbox: SlideBBox) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    x1 = max(0, min(w - 1, bbox.x1))
    y1 = max(0, min(h - 1, bbox.y1))
    x2 = max(1, min(w, bbox.x2))
    y2 = max(1, min(h, bbox.y2))
    crop = frame_bgr[y1:y2, x1:x2]
    return crop if crop.size else frame_bgr


def preprocess_for_ssim(crop_bgr: np.ndarray, size: Tuple[int, int] = (320, 180)) -> np.ndarray:
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return gray.astype(np.float32) / 255.0


def ssim_simple(a: np.ndarray, b: np.ndarray) -> float:
    """
    Lightweight SSIM (global). Good enough for slide-change detection.
    """
    C1 = (0.01 ** 2)
    C2 = (0.03 ** 2)

    mu_a = float(a.mean())
    mu_b = float(b.mean())
    var_a = float(((a - mu_a) ** 2).mean())
    var_b = float(((b - mu_b) ** 2).mean())
    cov = float(((a - mu_a) * (b - mu_b)).mean())

    num = (2 * mu_a * mu_b + C1) * (2 * cov + C2)
    den = (mu_a**2 + mu_b**2 + C1) * (var_a + var_b + C2)
    return float(num / den) if den != 0 else 0.0


def detect_transitions_and_segments(
    video_path: str,
    model_path: Optional[str] = None,
    config_path: Optional[str] = None,
    *,
    sample_every_sec: float = 1.0,
    downscale_max_side: int = 960,
    score_thresh: float = 0.95,
    ssim_thresh: float = 0.85,
    debounce_sec: float = 1.0,
    min_segment_sec: float = 2.0,
    predictor: Optional[Any] = None,
    use_full_frame_if_no_predictor: bool = False,
) -> Dict[str, Any]:
    """
    Returns:
      samples: [{t,bbox_xyxy,score,ssim}]
      transitions: [t...]
      segments: [{slide_id,start_time,end_time,duration}]
    """
    video_path = str(Path(video_path).expanduser().resolve())
    
    if predictor is None and model_path and config_path:
        predictor = build_predictor(model_path, config_path, score_thresh=score_thresh)
    elif predictor is None and not use_full_frame_if_no_predictor:
        raise ValueError("Provide predictor or model/config paths for slide detection.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = (frame_count / fps) if fps > 0 else 0.0

    t = 0.0
    prev_img: Optional[np.ndarray] = None
    last_transition = -debounce_sec

    samples: List[Dict[str, Any]] = []
    transitions: List[float] = []

    while t < duration:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        bbox = None
        if predictor is not None:
            bbox = detect_bbox_on_frame(frame, predictor, downscale_max_side=downscale_max_side)
            if bbox is None:
                samples.append({"t": t, "bbox_xyxy": None, "score": None, "ssim": None})
                prev_img = None  # reset comparison if detection missing
                t += sample_every_sec
                continue

        crop = crop_slide(frame, bbox) if bbox is not None else frame
        cur = preprocess_for_ssim(crop)

        ssim_val = None
        if prev_img is not None:
            ssim_val = ssim_simple(prev_img, cur)
            if ssim_val < ssim_thresh and (t - last_transition) >= debounce_sec:
                transitions.append(t)
                last_transition = t

        samples.append(
            {
                "t": t,
                "bbox_xyxy": [bbox.x1, bbox.y1, bbox.x2, bbox.y2],
                "score": bbox.score,
                "ssim": ssim_val,
            }
        )

        prev_img = cur
        t += sample_every_sec

    cap.release()

    # segments from boundaries
    boundaries = [0.0] + transitions + [duration]
    segments: List[Dict[str, Any]] = []
    slide_id = 1
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        if (end - start) >= min_segment_sec:
            segments.append(
                {"slide_id": slide_id, "start_time": start, "end_time": end, "duration": end - start}
            )
            slide_id += 1

    return {
        "video": {"path": video_path, "fps": fps, "duration": duration},
        "samples": samples,
        "transitions": transitions,
        "segments": segments,
        "params": {
            "sample_every_sec": sample_every_sec,
            "downscale_max_side": downscale_max_side,
            "score_thresh": score_thresh,
            "ssim_thresh": ssim_thresh,
            "debounce_sec": debounce_sec,
            "min_segment_sec": min_segment_sec,
        },
    }
