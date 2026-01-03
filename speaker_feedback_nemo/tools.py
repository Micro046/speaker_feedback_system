from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_payload(payload_path: str) -> Dict[str, Any]:
    path = Path(payload_path).expanduser()
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _trim(text: Optional[str], max_chars: int) -> str:
    text = (text or "").strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max_chars].rstrip() + "..."


def _timeline_summary(timeline: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not timeline:
        return {
            "total_slides": 0,
            "total_duration_sec": 0.0,
            "avg_slide_duration_sec": 0.0,
        }

    total_duration = max(float(seg.get("end_time", 0.0)) for seg in timeline)
    avg_duration = total_duration / max(1, len(timeline))
    return {
        "total_slides": len(timeline),
        "total_duration_sec": round(total_duration, 2),
        "avg_slide_duration_sec": round(avg_duration, 2),
    }


def slim_payload(payload: Dict[str, Any], *, max_slide_text_chars: int = 800) -> Dict[str, Any]:
    timeline = payload.get("timeline", []) or []
    slim_timeline: List[Dict[str, Any]] = []

    for seg in timeline:
        slim_timeline.append({
            "slide_id": seg.get("slide_id"),
            "start_time": seg.get("start_time"),
            "end_time": seg.get("end_time"),
            "duration": seg.get("duration"),
            "visual_word_count": seg.get("visual_word_count"),
            "visual_text": _trim(seg.get("visual_text"), max_slide_text_chars),
            "spoken_text": _trim(seg.get("spoken_text"), max_slide_text_chars),
            "ocr_parsed": seg.get("ocr_parsed", {}),
            "dominant_emotion": seg.get("dominant_emotion"),
            "emotion_confidence": seg.get("emotion_confidence"),
        })

    out: Dict[str, Any] = {
        "meta": payload.get("meta", {}),
        "speech_stats": payload.get("speech_stats", {}),
        "timeline": slim_timeline,
        "timeline_summary": _timeline_summary(timeline),
    }

    for key in ("clothing_analysis", "emotion_analysis", "gaze_analysis", "gesture_analysis"):
        if key in payload:
            out[key] = payload.get(key, {})

    return out


def get_slide_context(
    payload: Dict[str, Any],
    slide_id: int,
    *,
    max_slide_text_chars: int = 800,
) -> Dict[str, Any]:
    sid = str(slide_id)
    timeline = payload.get("timeline", []) or []
    target = None
    for seg in timeline:
        if str(seg.get("slide_id")) == sid:
            target = seg
            break

    if target is None:
        return {"slide_id": slide_id, "error": "Slide not found"}

    emotion_slide = (payload.get("emotion_analysis", {}) or {}).get("slide_summaries", {}) or {}
    gaze_slide = (payload.get("gaze_analysis", {}) or {}).get("slide_summaries", {}) or {}
    gesture_slide = (payload.get("gesture_analysis", {}) or {}).get("slide_summaries", {}) or {}
    gesture_issues = (payload.get("gesture_analysis", {}) or {}).get("issues", {}) or {}

    return {
        "slide_id": target.get("slide_id"),
        "time_range": {
            "start_time": target.get("start_time"),
            "end_time": target.get("end_time"),
            "duration": target.get("duration"),
        },
        "visual_text": _trim(target.get("visual_text"), max_slide_text_chars),
        "spoken_text": _trim(target.get("spoken_text"), max_slide_text_chars),
        "visual_word_count": target.get("visual_word_count"),
        "ocr_parsed": target.get("ocr_parsed", {}),
        "emotion": emotion_slide.get(sid, {}),
        "gaze": gaze_slide.get(sid, {}),
        "gesture": gesture_slide.get(sid, {}),
        "gesture_issues": gesture_issues.get(sid, []),
    }
