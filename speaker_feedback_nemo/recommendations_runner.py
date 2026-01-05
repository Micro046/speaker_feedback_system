# speaker_feedback_nemo/recommendations_runner.py
from __future__ import annotations

import argparse
import json
import os
import textwrap
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from agents.tools.recommendation_tool import run_nemo_react_recommendations
from speaker_feedback_nemo.tools import build_react_user_input


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    cleaned = [v for v in values if isinstance(v, (int, float))]
    if not cleaned:
        return None
    return sum(cleaned) / len(cleaned)


def _safe_ratio(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if not isinstance(numerator, (int, float)) or not isinstance(denominator, (int, float)):
        return None
    if denominator == 0:
        return None
    return numerator / denominator


def _excerpt(text: Optional[str], max_words: int = 24) -> str:
    if not text:
        return ""
    words = text.split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]) + "..."


def _normalize_output(text: str) -> str:
    if not text:
        return ""
    lines = [line.rstrip() for line in text.splitlines()]
    for i, line in enumerate(lines):
        if line.lstrip().startswith("# Presentation Feedback Report"):
            lines = lines[i:]
            break
    drop_prefixes = (
        "Parsing LLM output produced",
        "Thought:",
        "Action:",
        "Action Input:",
        "Observation:",
    )
    while lines and lines[0].startswith(drop_prefixes):
        lines.pop(0)
    while lines and lines[0].startswith("Thought:"):
        lines.pop(0)
    if lines and lines[0].startswith("Final Answer:"):
        first = lines.pop(0).replace("Final Answer:", "", 1).strip()
        if first:
            lines.insert(0, first)
    for i, line in enumerate(lines):
        if line.startswith("Final Answer:"):
            lines[i] = line.replace("Final Answer:", "", 1).lstrip()
    return "\n".join(lines).strip()


def load_payload(payload_path: str | Path) -> Dict[str, Any]:
    path = Path(payload_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Payload not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_recommendation_payload(raw_payload: Dict[str, Any]) -> Dict[str, Any]:
    segments = raw_payload.get("segments") or []
    slide_time_map: Dict[str, Dict[str, float]] = {}
    for seg in segments:
        slide_id = seg.get("slide_id")
        start = seg.get("start_time")
        end = seg.get("end_time")
        if slide_id is None or not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            continue
        slide_time_map[str(slide_id)] = {"start_time": start, "end_time": end}

    slide_summaries = []
    emotion = raw_payload.get("emotion_analysis") or {}
    emotion_slides = emotion.get("slide_summaries") or {}
    gaze = raw_payload.get("gaze_analysis") or {}
    gaze_slides = gaze.get("slide_summaries") or {}
    gesture = raw_payload.get("gesture_analysis") or {}
    gesture_slides = gesture.get("slide_summaries") or {}

    for seg in segments:
        slide_id = seg.get("slide_id")
        if slide_id is None:
            continue
        slide_key = str(slide_id)
        summary = {
            "slide_id": slide_id,
            "start_time": seg.get("start_time"),
            "end_time": seg.get("end_time"),
            "duration_sec": seg.get("duration_sec"),
            "visual_text_excerpt": _excerpt(seg.get("visual_text")),
            "spoken_text_excerpt": _excerpt(seg.get("spoken_text")),
            "visual_word_count": seg.get("visual_word_count"),
            "spoken_word_count": seg.get("spoken_word_count"),
            "speech_coverage_ratio": seg.get("speech_coverage_ratio"),
            "speech_overlap_sec": seg.get("speech_overlap_sec"),
            "dominant_emotion": seg.get("dominant_emotion"),
            "emotion_confidence": seg.get("emotion_confidence"),
        }

        emotion_slide = emotion_slides.get(slide_key)
        if isinstance(emotion_slide, dict):
            summary["emotion_summary"] = {
                "dominant_emotion": emotion_slide.get("dominant_emotion"),
                "avg_confidence": emotion_slide.get("avg_confidence"),
                "coverage_ratio": emotion_slide.get("coverage_ratio"),
                "emotion_frequency": emotion_slide.get("emotion_frequency"),
            }

        gaze_slide = gaze_slides.get(slide_key)
        if isinstance(gaze_slide, dict):
            summary["gaze_summary"] = {
                "dominant_gaze": gaze_slide.get("dominant_gaze"),
                "valid_gaze_ratio": gaze_slide.get("valid_gaze_ratio"),
                "distribution": gaze_slide.get("distribution"),
                "avg_gaze_confidence": gaze_slide.get("avg_gaze_confidence"),
                "evidence_frames": (gaze_slide.get("evidence_frames") or [])[:2],
            }

        gesture_slide = gesture_slides.get(slide_key)
        if isinstance(gesture_slide, dict):
            summary["gesture_summary"] = {
                "pose_coverage": gesture_slide.get("pose_coverage"),
                "recommendations": gesture_slide.get("recommendations"),
                "evidence_frames": (gesture_slide.get("evidence_frames") or [])[:2],
            }

        slide_summaries.append(summary)

    speech_coverage = [seg.get("speech_coverage_ratio") for seg in segments]
    spoken_words = [seg.get("spoken_word_count") for seg in segments]
    visual_words = [seg.get("visual_word_count") for seg in segments]
    durations = [seg.get("duration_sec") for seg in segments]
    speech_overlap = [seg.get("speech_overlap_sec") for seg in segments]
    speech_to_visual_ratios = [
        _safe_ratio(seg.get("spoken_word_count"), seg.get("visual_word_count")) for seg in segments
    ]
    wpm_by_slide = {}
    overlap_ratio_by_slide = {}
    spoken_to_visual_by_slide = {}
    for seg in segments:
        slide_id = seg.get("slide_id")
        duration = seg.get("duration_sec")
        spoken_count = seg.get("spoken_word_count")
        visual_count = seg.get("visual_word_count")
        overlap_sec = seg.get("speech_overlap_sec")
        if slide_id is None:
            continue
        if isinstance(spoken_count, (int, float)) and isinstance(duration, (int, float)) and duration > 0:
            wpm_by_slide[str(slide_id)] = (spoken_count / duration) * 60
        if isinstance(overlap_sec, (int, float)) and isinstance(duration, (int, float)) and duration > 0:
            overlap_ratio_by_slide[str(slide_id)] = overlap_sec / duration
        ratio = _safe_ratio(spoken_count, visual_count)
        if ratio is not None:
            spoken_to_visual_by_slide[str(slide_id)] = ratio

    emotion = raw_payload.get("emotion_analysis") or {}
    emotion_slides = emotion.get("slide_summaries") or {}
    emotion_confidences = [
        slide.get("avg_confidence") for slide in emotion_slides.values() if isinstance(slide, dict)
    ]
    emotion_coverages = [
        slide.get("coverage_ratio") for slide in emotion_slides.values() if isinstance(slide, dict)
    ]

    gaze = raw_payload.get("gaze_analysis") or {}
    gaze_overall = gaze.get("overall_summary") or {}

    gesture = raw_payload.get("gesture_analysis") or {}
    gesture_overall = gesture.get("overall") or {}

    clothing = raw_payload.get("clothing_analysis") or {}
    clothing_coverage = clothing.get("coverage") or {}

    total_duration = sum([v for v in durations if isinstance(v, (int, float))]) if durations else None

    derived_metrics: Dict[str, Any] = {
        "slides_total": len(segments),
        "total_duration_sec": total_duration,
        "total_spoken_word_count": sum([v for v in spoken_words if isinstance(v, (int, float))]) if spoken_words else None,
        "total_visual_word_count": sum([v for v in visual_words if isinstance(v, (int, float))]) if visual_words else None,
        "avg_speech_coverage_ratio": _mean(speech_coverage),
        "min_speech_coverage_ratio": min(speech_coverage) if speech_coverage else None,
        "max_speech_coverage_ratio": max(speech_coverage) if speech_coverage else None,
        "total_speech_overlap_sec": sum([v for v in speech_overlap if isinstance(v, (int, float))]) if speech_overlap else None,
        "avg_speech_overlap_ratio": _mean([v for v in overlap_ratio_by_slide.values()]),
        "avg_spoken_word_count": _mean(spoken_words),
        "avg_visual_word_count": _mean(visual_words),
        "avg_spoken_to_visual_word_ratio": _mean(speech_to_visual_ratios),
        "wpm_by_slide": wpm_by_slide,
        "spoken_to_visual_ratio_by_slide": spoken_to_visual_by_slide,
        "speech_overlap_ratio_by_slide": overlap_ratio_by_slide,
        "avg_slide_duration_sec": _mean(durations),
        "max_slide_duration_sec": max(durations) if durations else None,
        "emotion_avg_confidence": _mean(emotion_confidences),
        "emotion_avg_coverage_ratio": _mean(emotion_coverages),
        "gaze_valid_ratio": gaze_overall.get("valid_gaze_ratio"),
        "gaze_distribution": gaze_overall.get("percentages"),
        "pose_coverage": gesture_overall.get("pose_coverage"),
        "pose_frames_processed": gesture_overall.get("frames_processed"),
        "clothing_frames_used": clothing_coverage.get("frames_used"),
    }

    trimmed_payload: Dict[str, Any] = {}
    for key in ("video_info", "segments", "clothing_analysis", "emotion_analysis", "gaze_analysis", "gesture_analysis"):
        if key in raw_payload:
            trimmed_payload[key] = raw_payload[key]

    trimmed_payload["slide_time_map"] = slide_time_map
    trimmed_payload["slide_summaries"] = slide_summaries
    trimmed_payload["derived_metrics"] = derived_metrics
    return trimmed_payload


def generate_recommendations(
    *,
    payload_path: str | Path,
    constraints: Dict[str, str],
    config_path: str | Path = "speaker_feedback_nemo/configs/recommendations.yml",
    top_k: int = 6,
) -> str:
    raw_payload = load_payload(payload_path)
    recommendation_payload = build_recommendation_payload(raw_payload)
    user_input = build_react_user_input(
        payload=recommendation_payload,
        constraints=constraints,
        top_k_recommendations=top_k,
    )
    output = run_nemo_react_recommendations(
        config_file=str(Path(config_path).expanduser().resolve()),
        user_input=user_input,
    )
    return _normalize_output(output)


def write_markdown_report(text: str, output_path: str | Path) -> Path:
    path = Path(output_path).expanduser().resolve()
    path.write_text(text, encoding="utf-8")
    return path


def write_pdf_report(text: str, output_path: str | Path) -> Path:
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
    except ImportError as exc:
        raise RuntimeError("reportlab is required to write PDF output.") from exc

    path = Path(output_path).expanduser().resolve()
    page_width, page_height = letter
    margin = 72
    x = margin
    y = page_height - margin
    line_height = 12

    c = canvas.Canvas(str(path), pagesize=letter)
    for raw_line in text.splitlines():
        if not raw_line.strip():
            y -= line_height
            if y < margin:
                c.showPage()
                y = page_height - margin
            continue
        for line in textwrap.wrap(raw_line, width=100):
            if y < margin:
                c.showPage()
                y = page_height - margin
            c.drawString(x, y, line)
            y -= line_height
    c.save()
    return path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NeMo ReAct recommendations.")
    parser.add_argument(
        "--payload",
        default="outputs/analysis_payload.json",
        help="Path to analysis_payload.json.",
    )
    parser.add_argument(
        "--config",
        default="speaker_feedback_nemo/configs/recommendations.yml",
        help="Path to NAT recommendations config.",
    )
    parser.add_argument("--presentation-type", default="", help="Presentation type.")
    parser.add_argument("--audience", default="", help="Audience.")
    parser.add_argument("--goal", default="", help="Goal.")
    parser.add_argument("--time-limit", default="", help="Time limit.")
    parser.add_argument("--top-k", type=int, default=6, help="Number of recommendations.")
    parser.add_argument("--output", default="", help="Optional path to write recommendations.")
    parser.add_argument("--markdown-out", default="", help="Optional path to write a markdown report.")
    parser.add_argument("--pdf-out", default="", help="Optional path to write a PDF report (requires reportlab).")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise EnvironmentError("OPENROUTER_API_KEY is not set.")

    constraints = {
        "presentation_type": args.presentation_type,
        "audience": args.audience,
        "goal": args.goal,
        "time_limit": args.time_limit,
    }
    recommendations = generate_recommendations(
        payload_path=args.payload,
        constraints=constraints,
        config_path=args.config,
        top_k=args.top_k,
    )

    if args.output:
        Path(args.output).expanduser().resolve().write_text(recommendations, encoding="utf-8")
    if args.markdown_out:
        write_markdown_report(recommendations, args.markdown_out)
    if args.pdf_out:
        write_pdf_report(recommendations, args.pdf_out)
    print(recommendations)


if __name__ == "__main__":
    main()
