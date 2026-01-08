# speaker_feedback/nemo/recommendations_runner.py
from __future__ import annotations

import argparse
import json
import os
import re
import string
import textwrap
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

from speaker_feedback.agents.tools.recommendation_tool import run_nemo_react_recommendations
from speaker_feedback.nemo.tools import (
    build_react_user_input,
    build_storytelling_input,
    build_visual_coaching_input,
)

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "recommendations.yml"

_WORD_RE = re.compile(r"[A-Za-z]+")
_TOKEN_RE = re.compile(r"[A-Za-z]+")

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "has",
    "have",
    "he",
    "her",
    "his",
    "i",
    "in",
    "is",
    "it",
    "its",
    "me",
    "my",
    "not",
    "of",
    "on",
    "or",
    "our",
    "she",
    "so",
    "that",
    "the",
    "their",
    "them",
    "there",
    "these",
    "they",
    "this",
    "to",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "will",
    "with",
    "you",
    "your",
}


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


def _keywords(text: Optional[str]) -> set[str]:
    if not text:
        return set()
    text = text.replace("\u2019", "'").lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = _TOKEN_RE.findall(text)
    tokens = [t for t in tokens if t and t not in _STOPWORDS and len(t) > 2 and _WORD_RE.search(t)]
    return set(tokens)


def _jaccard(a: set[str], b: set[str]) -> Optional[float]:
    if not a and not b:
        return None
    union = a | b
    if not union:
        return None
    return len(a & b) / len(union)


def _aggregate_gaze_distribution(
    gaze_slides: Dict[str, Any],
) -> Tuple[Optional[Dict[str, float]], Optional[float]]:
    total_valid = 0.0
    total_sampled = 0.0
    left = right = center = 0.0

    for slide_summary in gaze_slides.values():
        if not isinstance(slide_summary, dict):
            continue
        sampled = slide_summary.get("sampled_frames")
        valid = slide_summary.get("valid_frames")
        distribution = slide_summary.get("distribution") or {}
        if not isinstance(sampled, (int, float)) or sampled <= 0:
            continue
        if not isinstance(valid, (int, float)):
            coverage = slide_summary.get("coverage_ratio")
            valid = sampled * float(coverage) if isinstance(coverage, (int, float)) else 0.0

        total_sampled += float(sampled)
        total_valid += float(valid)

        left += float(valid) * (float(distribution.get("left", 0.0)) / 100.0)
        right += float(valid) * (float(distribution.get("right", 0.0)) / 100.0)
        center += float(valid) * (float(distribution.get("center", 0.0)) / 100.0)

    if total_valid <= 0 or total_sampled <= 0:
        return None, None

    distribution = {
        "left": round(100.0 * left / total_valid, 1),
        "right": round(100.0 * right / total_valid, 1),
        "center": round(100.0 * center / total_valid, 1),
    }
    valid_ratio = round(total_valid / total_sampled, 3)
    return distribution, valid_ratio


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


def _compact_dict(data: Dict[str, Any], keys: Iterable[str]) -> Dict[str, Any]:
    return {k: data[k] for k in keys if k in data and data[k] is not None}


def _trim_visual_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return payload

    overall_in = payload.get("overall") or {}
    out: Dict[str, Any] = {}
    overall_out: Dict[str, Any] = {}

    clothing_in = overall_in.get("clothing") or payload.get("clothing") or payload.get("clothing_analysis")
    if isinstance(clothing_in, dict):
        clothing = _compact_dict(
            clothing_in,
            (
                "summary",
                "description",
                "style",
                "top",
                "is_appropriate",
                "recommendation",
                "detected_attributes",
                "coverage",
            ),
        )
        coverage = clothing.get("coverage")
        if isinstance(coverage, dict):
            clothing["coverage"] = _compact_dict(coverage, ("slides_with_samples", "frames_used"))
        overall_out["clothing"] = clothing

    gaze_in = overall_in.get("gaze") or payload.get("gaze") or payload.get("gaze_analysis")
    if isinstance(gaze_in, dict):
        gaze_summary = gaze_in.get("overall_summary") or gaze_in.get("overall")
        gaze = {}
        if isinstance(gaze_summary, dict):
            gaze["overall_summary"] = gaze_summary
        overall_out["gaze"] = gaze

    emotion_in = overall_in.get("emotion") or payload.get("emotion") or payload.get("emotion_analysis")
    if isinstance(emotion_in, dict):
        emotion_summary = emotion_in.get("overall_stats") or emotion_in.get("overall")
        emotion = {}
        if isinstance(emotion_summary, dict):
            emotion["overall_stats"] = emotion_summary
        overall_out["emotion"] = emotion

    gesture_in = overall_in.get("gesture") or payload.get("gesture") or payload.get("gesture_analysis")
    if isinstance(gesture_in, dict):
        gesture_overall = gesture_in.get("overall") or {}
        gesture = {}
        if isinstance(gesture_overall, dict):
            joint_stats = dict(gesture_overall.get("joint_statistics") or {})
            joint_stats.pop("frame_states", None)
            gesture["overall"] = dict(gesture_overall, joint_statistics=joint_stats)
        overall_out["gesture"] = gesture

    if overall_out:
        out["overall"] = overall_out

    events_in = payload.get("events")
    if isinstance(events_in, dict):
        events_out: Dict[str, Any] = {}
        if "multimodal_events" in events_in:
            events_out["multimodal_events"] = events_in["multimodal_events"]

        gaze_by_slide = events_in.get("gaze_by_slide")
        if isinstance(gaze_by_slide, dict):
            trimmed = {}
            for sid, entry in gaze_by_slide.items():
                if not isinstance(entry, dict):
                    continue
                trimmed[sid] = _compact_dict(
                    entry,
                    (
                        "dominant_focus",
                        "focus_distribution",
                        "focus_dist",
                        "distribution",
                        "recommendation",
                        "sampled_frames",
                        "valid_frames",
                        "coverage_ratio",
                        "evidence_frames",
                        "data_quality",
                    ),
                )
            events_out["gaze_by_slide"] = trimmed

        emotion_by_slide = events_in.get("emotion_by_slide")
        if isinstance(emotion_by_slide, dict):
            trimmed = {}
            for sid, entry in emotion_by_slide.items():
                if not isinstance(entry, dict):
                    continue
                trimmed[sid] = _compact_dict(
                    entry,
                    (
                        "dominant_emotion",
                        "emotion_frequency",
                        "emotion_distribution",
                        "coverage_ratio",
                        "avg_confidence",
                        "avg_entropy",
                        "top_frames",
                    ),
                )
            events_out["emotion_by_slide"] = trimmed

        gesture_by_slide = events_in.get("gesture_by_slide")
        if isinstance(gesture_by_slide, dict):
            trimmed = {}
            for sid, entry in gesture_by_slide.items():
                if not isinstance(entry, dict):
                    continue
                joint_stats = dict(entry.get("joint_statistics") or {})
                joint_stats.pop("frame_states", None)
                trimmed[sid] = _compact_dict(
                    dict(entry, joint_statistics=joint_stats),
                    (
                        "joint_statistics",
                        "recommendations",
                        "pose_coverage",
                        "frames_sampled",
                        "frames_processed",
                        "evidence_frames",
                        "events",
                        "data_quality",
                    ),
                )
            events_out["gesture_by_slide"] = trimmed

        if events_out:
            out["events"] = events_out

    return out if out else payload


def _trim_story_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return payload

    out: Dict[str, Any] = {}
    overall = payload.get("overall")
    if isinstance(overall, dict):
        out["overall"] = overall

    derived_metrics = payload.get("derived_metrics")
    if isinstance(derived_metrics, dict):
        out["derived_metrics"] = derived_metrics

    slides_in = payload.get("slides")
    if isinstance(slides_in, list):
        slides_out = []
        for slide in slides_in:
            if not isinstance(slide, dict):
                continue
            speech = slide.get("speech") or {}
            trimmed_speech = _compact_dict(
                speech,
                (
                    "wpm",
                    "intelligibility",
                    "filler_word_count",
                    "filler_phrase_count",
                    "noise_fraction",
                    "speech_text_preview",
                    "evidence_intervals",
                ),
            )
            content_alignment = slide.get("content_alignment") or {}
            trimmed_alignment = {}
            if isinstance(content_alignment, dict):
                trimmed_alignment = _compact_dict(content_alignment, ("similarity", "ocr_word_count"))

            slides_out.append(
                _compact_dict(
                    {
                        "slide_id": slide.get("slide_id"),
                        "start_time": slide.get("start_time"),
                        "end_time": slide.get("end_time"),
                        "duration": slide.get("duration"),
                        "ocr_text": slide.get("ocr_text"),
                        "description": slide.get("description"),
                        "speech": trimmed_speech,
                        "content_alignment": trimmed_alignment,
                    },
                    (
                        "slide_id",
                        "start_time",
                        "end_time",
                        "duration",
                        "ocr_text",
                        "description",
                        "speech",
                        "content_alignment",
                    ),
                )
            )
        out["slides"] = slides_out

    if out:
        return out
    return payload


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
    segments_compact = []
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
        visual_text = seg.get("visual_text") or ""
        spoken_text = seg.get("spoken_text") or ""
        visual_kw = _keywords(visual_text)
        spoken_kw = _keywords(spoken_text)
        overlap_jaccard = _jaccard(visual_kw, spoken_kw)
        visual_kw_coverage = (len(visual_kw & spoken_kw) / len(visual_kw)) if visual_kw else None
        spoken_kw_coverage = (len(visual_kw & spoken_kw) / len(spoken_kw)) if spoken_kw else None

        summary = {
            "slide_id": slide_id,
            "start_time": seg.get("start_time"),
            "end_time": seg.get("end_time"),
            "duration_sec": seg.get("duration_sec"),
            "visual_text_excerpt": _excerpt(visual_text),
            "spoken_text_excerpt": _excerpt(spoken_text),
            "visual_word_count": seg.get("visual_word_count"),
            "spoken_word_count": seg.get("spoken_word_count"),
            "speech_coverage_ratio": seg.get("speech_coverage_ratio"),
            "speech_overlap_sec": seg.get("speech_overlap_sec"),
            "dominant_emotion": seg.get("dominant_emotion"),
            "emotion_confidence": seg.get("emotion_confidence"),
            "content_overlap_jaccard": overlap_jaccard,
            "visual_keyword_coverage": visual_kw_coverage,
            "spoken_keyword_coverage": spoken_kw_coverage,
        }

        slide_summaries.append(summary)
        segments_compact.append(
            {
                "slide_id": slide_id,
                "start_time": seg.get("start_time"),
                "end_time": seg.get("end_time"),
                "duration_sec": seg.get("duration_sec"),
                "visual_text_excerpt": summary["visual_text_excerpt"],
                "visual_word_count": seg.get("visual_word_count"),
                "spoken_text_excerpt": summary["spoken_text_excerpt"],
                "spoken_word_count": seg.get("spoken_word_count"),
                "speech_overlap_sec": seg.get("speech_overlap_sec"),
                "speech_coverage_ratio": seg.get("speech_coverage_ratio"),
                "content_overlap_jaccard": overlap_jaccard,
                "visual_keyword_coverage": visual_kw_coverage,
                "spoken_keyword_coverage": spoken_kw_coverage,
            }
        )

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
    gaze_distribution_weighted, gaze_valid_ratio_weighted = _aggregate_gaze_distribution(gaze_slides)

    gesture = raw_payload.get("gesture_analysis") or {}
    gesture_overall = gesture.get("overall") or {}

    clothing = raw_payload.get("clothing_analysis") or {}
    clothing_coverage = clothing.get("coverage") or {}

    start_times = [seg.get("start_time") for seg in segments if isinstance(seg.get("start_time"), (int, float))]
    end_times = [seg.get("end_time") for seg in segments if isinstance(seg.get("end_time"), (int, float))]
    if start_times and end_times:
        total_duration = max(end_times) - min(start_times)
    else:
        total_duration = sum([v for v in durations if isinstance(v, (int, float))]) if durations else None

    jaccards = [s.get("content_overlap_jaccard") for s in slide_summaries]
    visual_kw_coverages = [s.get("visual_keyword_coverage") for s in slide_summaries]
    spoken_kw_coverages = [s.get("spoken_keyword_coverage") for s in slide_summaries]

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
        "avg_content_overlap_jaccard": _mean(jaccards),
        "avg_visual_keyword_coverage": _mean(visual_kw_coverages),
        "avg_spoken_keyword_coverage": _mean(spoken_kw_coverages),
        "avg_slide_duration_sec": _mean(durations),
        "max_slide_duration_sec": max(durations) if durations else None,
        "emotion_avg_confidence": _mean(emotion_confidences),
        "emotion_avg_coverage_ratio": _mean(emotion_coverages),
        "gaze_valid_ratio": gaze_valid_ratio_weighted if gaze_valid_ratio_weighted is not None else gaze_overall.get("valid_gaze_ratio"),
        "gaze_distribution": gaze_distribution_weighted if gaze_distribution_weighted is not None else gaze_overall.get("percentages"),
        "pose_coverage": gesture_overall.get("pose_coverage"),
        "pose_frames_processed": gesture_overall.get("frames_processed"),
        "clothing_frames_used": clothing_coverage.get("frames_used"),
    }

    trimmed_payload: Dict[str, Any] = {}
    for key in (
        "video_info",
        "speech_analysis",
        "clothing_analysis",
        "emotion_analysis",
        "gaze_analysis",
        "gesture_analysis",
        "policy",
        "presentation_context",
        "thresholds",
    ):
        if key in raw_payload:
            trimmed_payload[key] = raw_payload[key]

    trimmed_payload["segments"] = segments_compact
    trimmed_payload["slide_time_map"] = slide_time_map
    trimmed_payload["slide_summaries"] = slide_summaries
    trimmed_payload["derived_metrics"] = derived_metrics
    return trimmed_payload


def generate_recommendations(
    *,
    payload_path: str | Path,
    constraints: Dict[str, str],
    config_path: str | Path = _DEFAULT_CONFIG_PATH,
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


def generate_storytelling_report(
    *,
    payload_path: str | Path,
    constraints: Dict[str, str],
    config_path: str | Path = _DEFAULT_CONFIG_PATH,
    top_k: int = 6,
) -> str:
    raw_payload = load_payload(payload_path)
    trimmed_payload = _trim_story_payload(raw_payload)
    user_input = build_storytelling_input(
        payload=trimmed_payload,
        constraints=constraints,
        top_k_recommendations=top_k,
    )
    output = run_nemo_react_recommendations(
        config_file=str(Path(config_path).expanduser().resolve()),
        user_input=user_input,
    )
    return _normalize_output(output)


def generate_visual_coaching_report(
    *,
    payload_path: str | Path,
    constraints: Dict[str, str],
    config_path: str | Path = _DEFAULT_CONFIG_PATH,
    top_k: int = 6,
) -> str:
    raw_payload = load_payload(payload_path)
    trimmed_payload = _trim_visual_payload(raw_payload)
    user_input = build_visual_coaching_input(
        payload=trimmed_payload,
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
        default="data/outputs/analysis_payload.json",
        help="Path to analysis_payload.json.",
    )
    parser.add_argument(
        "--config",
        default=str(_DEFAULT_CONFIG_PATH),
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
