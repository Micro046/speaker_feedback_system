from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, Tuple

import cv2
import mediapipe as mp


@dataclass
class GazeHeuristics:
    center_thresh: float = 0.02
    max_delta: float = 0.06


class MediaPipeGazeDirection:
    """
    Lightweight gaze direction estimator using MediaPipe FaceMesh.
    This is a coarse head/gaze proxy (left/center/right) that is robust
    enough for aggregate feedback.
    """

    def __init__(
        self,
        *,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        heuristics: GazeHeuristics | None = None,
    ):
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            refine_landmarks=False,
            max_num_faces=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._cfg = heuristics or GazeHeuristics()

        # Face mesh landmark indices for eye corners + nose tip.
        self._left_eye = (33, 133)
        self._right_eye = (362, 263)
        self._nose_tip = 1

    def process_frame(self, face_crop_bgr) -> Tuple[str, float]:
        rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
        res = self._mesh.process(rgb)
        if not res.multi_face_landmarks:
            return "no_face", 0.0

        lm = res.multi_face_landmarks[0].landmark
        lx = (lm[self._left_eye[0]].x + lm[self._left_eye[1]].x) / 2.0
        rx = (lm[self._right_eye[0]].x + lm[self._right_eye[1]].x) / 2.0
        eye_center_x = (lx + rx) / 2.0
        nose_x = lm[self._nose_tip].x

        dx = float(nose_x - eye_center_x)
        if abs(dx) <= self._cfg.center_thresh:
            label = "center"
        elif dx > 0:
            label = "right"
        else:
            label = "left"

        conf = min(1.0, abs(dx) / max(self._cfg.max_delta, 1e-6))
        return label, float(conf)

    @staticmethod
    def summarize_gaze(labels) -> Dict[str, object]:
        valid = [l for l in labels if l in {"left", "right", "center"}]
        total = len(labels)
        valid_total = len(valid)

        counts = Counter(valid)
        most_common_valid = counts.most_common(1)[0][0] if counts else "no_gaze"
        percentages = {
            "left": round(100.0 * counts.get("left", 0) / max(valid_total, 1), 1),
            "right": round(100.0 * counts.get("right", 0) / max(valid_total, 1), 1),
            "center": round(100.0 * counts.get("center", 0) / max(valid_total, 1), 1),
        }
        valid_ratio = round(valid_total / max(total, 1), 3)

        if valid_ratio < 0.3:
            rec = "Limited reliable gaze frames. Improve lighting and face visibility."
        elif percentages["left"] > 40:
            rec = "Frequent left gaze. Try to re-center toward the camera."
        elif percentages["right"] > 40:
            rec = "Frequent right gaze. Try to re-center toward the camera."
        else:
            rec = "Eye contact looks balanced."

        return {
            "most_common_valid": most_common_valid,
            "valid_gaze_ratio": valid_ratio,
            "percentages": percentages,
            "recommendation": rec,
        }

    def close(self) -> None:
        if self._mesh:
            self._mesh.close()
            self._mesh = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
