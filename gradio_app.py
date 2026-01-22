from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import gradio as gr
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env", override=True)
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DEFAULT_DETECTRON_MODEL_PATH = Path("/datasets/model_best/model_best.pth")
DEFAULT_DETECTRON_CONFIG_PATH = Path("/notebooks/data/cache/my_custom_config.yaml")
DEFAULT_OCR_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

from emotiefflib.facial_analysis import EmotiEffLibRecognizer, get_model_list  # noqa: E402
from speaker_feedback.agents.tools.clothing_tool import clothing_analysis_tool  # noqa: E402
from speaker_feedback.agents.tools.face_cache_tools import build_face_cache_tool  # noqa: E402
from speaker_feedback.agents.tools.gaze_tool import gaze_analysis_tool  # noqa: E402
from speaker_feedback.agents.tools.emotion_tool import emotion_analysis_tool  # noqa: E402
from speaker_feedback.agents.tools.slide_ocr_tool import slide_extraction_tool  # noqa: E402
from speaker_feedback.agents.tools.speech_analysis_tool import analyze_speech_tool  # noqa: E402
from speaker_feedback.config.thresholds import DEFAULT_THRESHOLDS_PATH, load_thresholds  # noqa: E402
from speaker_feedback.reporting.combined_recommendations import (  # noqa: E402
    ContentRunConfig,
    VisualRunConfig,
    run_combined_recommendations,
)
from speaker_feedback.video_analysis.clothing_analysis import ClothesCLIP  # noqa: E402
from speaker_feedback.video_analysis.gaze_analysis import GazeHeuristics, MediaPipeGazeDirection  # noqa: E402
from speaker_feedback.video_analysis.gesture_analysis import (  # noqa: E402
    GestureHeuristics,
    Gestures,
    gesture_analysis_tool,
)
import torch  # noqa: E402

PAPER_GRADIENT_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

body {
  background:
    radial-gradient(900px 300px at 15% 0%, rgba(255, 255, 255, 0.85), rgba(255, 255, 255, 0)),
    radial-gradient(800px 280px at 85% 10%, rgba(205, 231, 244, 0.6), rgba(255, 255, 255, 0)),
    linear-gradient(135deg, #f7f5f0 0%, #e6eef5 100%);
}
.gradio-container {
  background: transparent;
  font-family: "Space Grotesk", system-ui, -apple-system, sans-serif;
}
.prism-hero {
  padding: 24px 26px;
  border-radius: 18px;
  background: linear-gradient(135deg, rgba(255, 255, 255, 0.95), rgba(239, 246, 250, 0.92));
  border: 1px solid rgba(34, 62, 81, 0.08);
  box-shadow: 0 12px 30px rgba(36, 60, 78, 0.12);
}
.prism-title {
  font-family: "Fraunces", "Times New Roman", serif;
  font-size: 30px;
  margin: 0 0 6px 0;
  color: #1b2b38;
}
.prism-subtitle {
  margin: 0;
  color: #3c5466;
  font-size: 15px;
}
.prism-badges {
  display: flex;
  gap: 8px;
  margin-top: 14px;
  flex-wrap: wrap;
}
.prism-badge {
  padding: 6px 10px;
  border-radius: 999px;
  background: #122631;
  color: #f3f7fa;
  font-size: 12px;
  letter-spacing: 0.3px;
}
.prism-grid {
  margin-top: 18px;
  gap: 14px;
}
.prism-panel {
  background: rgba(255, 255, 255, 0.92);
  border-radius: 16px;
  border: 1px solid rgba(25, 40, 56, 0.08);
  padding: 16px;
  box-shadow: 0 8px 18px rgba(28, 45, 59, 0.08);
}
.prism-side h3 {
  margin-top: 0;
}
.prism-step {
  padding: 10px 12px;
  border-radius: 12px;
  background: #f4f8fb;
  border: 1px dashed rgba(38, 64, 82, 0.18);
  font-size: 13px;
  margin-bottom: 10px;
}
.prism-upload {
  border-radius: 14px;
}
.prism-primary button {
  background: linear-gradient(135deg, #1b2b38, #33506a);
  border: none;
  color: #f8fbff;
  font-weight: 600;
  border-radius: 999px;
  padding: 10px 18px;
  box-shadow: 0 10px 20px rgba(29, 48, 64, 0.24);
}
.prism-primary button:hover {
  filter: brightness(1.05);
}
.prism-log textarea {
  font-size: 12px;
  background: #0f1d27;
  color: #e5f1f7;
}
.prism-pdf iframe {
  border-radius: 16px;
  border: 1px solid rgba(20, 40, 55, 0.1);
}
"""


OFFLINE_MODE = os.getenv("SF_OFFLINE_MODE", "1") == "1"
if OFFLINE_MODE:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def _push_log(log_lines: List[str], message: str) -> str:
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {message}"
    print(line, flush=True)
    log_lines.append(line)
    return "\n".join(log_lines)


def _resolve_path(env_key: str, default_path: Path) -> str:
    value = os.getenv(env_key)
    path = Path(value) if value else default_path
    if not path.exists():
        raise RuntimeError(
            f"{env_key} not found. Set it in .env or place the file at {path}"
        )
    return str(path)


def _overlap_sec(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _weighted_avg(
    items: List[dict],
    value_key: str,
    start_key: str,
    end_key: str,
    seg_start: float,
    seg_end: float,
) -> Optional[float]:
    total = 0.0
    weight = 0.0
    for item in items:
        s = float(item.get(start_key, 0.0))
        e = float(item.get(end_key, s))
        o = _overlap_sec(s, e, seg_start, seg_end)
        if o > 0:
            total += o * float(item.get(value_key, 0.0))
            weight += o
    return (total / weight) if weight > 0 else None


def _text_for_interval(segments: List[dict], seg_start: float, seg_end: float, max_chars: int = 300) -> str:
    parts = []
    for seg in segments:
        s = float(seg.get("start", 0.0))
        e = float(seg.get("end", s))
        if _overlap_sec(s, e, seg_start, seg_end) > 0:
            parts.append(seg.get("text", "").strip())
    text = " ".join([p for p in parts if p])
    return text[:max_chars]


def _count_occurrences(occ_list: List[dict], seg_start: float, seg_end: float) -> int:
    count = 0
    for occ in occ_list:
        t = float(occ.get("start", 0.0))
        if seg_start <= t <= seg_end:
            count += 1
    return count


def _noise_fraction(noise_intervals: List[List[float]], seg_start: float, seg_end: float) -> float:
    dur = max(0.0, seg_end - seg_start)
    if dur == 0:
        return 0.0
    total = 0.0
    for a, b in noise_intervals:
        total += _overlap_sec(a, b, seg_start, seg_end)
    return total / dur


def _clip_intervals(intervals: List[List[float]], seg_start: float, seg_end: float) -> List[List[float]]:
    out = []
    for a, b in intervals:
        a = float(a)
        b = float(b)
        if _overlap_sec(a, b, seg_start, seg_end) > 0:
            out.append([max(a, seg_start), min(b, seg_end)])
    return out


def _token_set(text: str) -> set[str]:
    import re
    toks = re.findall(r"[A-Za-z0-9]+", (text or "").lower())
    return set([t for t in toks if t])


def _text_similarity(a: str, b: str) -> Dict[str, float]:
    from difflib import SequenceMatcher
    ta = _token_set(a)
    tb = _token_set(b)
    jaccard = (len(ta & tb) / len(ta | tb)) if (ta or tb) else 0.0
    edit = SequenceMatcher(None, (a or "").lower(), (b or "").lower()).ratio()
    return {
        "jaccard": round(float(jaccard), 3),
        "edit_ratio": round(float(edit), 3),
    }


def _text_word_count(text: str) -> int:
    return len([t for t in _token_set(text)])


def _label_counts(frames: List[dict], key: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for f in frames or []:
        label = f.get(key)
        if label:
            counts[label] = counts.get(label, 0) + 1
    return counts


def _build_payloads(
    *,
    speech_result: dict,
    slide_result: dict,
    clothing_result: dict | None,
    gaze_result: dict | None,
    emotion_result: dict | None,
    gesture_result: dict | None,
) -> tuple[dict, dict]:
    speech_segments = speech_result.get("segments", [])
    rate_windows = speech_result.get("speech_rate", {}).get("windows", [])
    intel_segments = speech_result.get("intelligibility", {}).get("per_segment", [])
    fillers_words = speech_result.get("filler_occurrences", {}).get("words", [])
    fillers_phrases = speech_result.get("filler_occurrences", {}).get("phrases", [])
    noise_intervals = speech_result.get("background_noise", {}).get("intervals", [])
    slow_intervals = speech_result.get("speech_rate", {}).get("slow", [])
    fast_intervals = speech_result.get("speech_rate", {}).get("fast", [])
    low_intel_intervals = speech_result.get("intelligibility", {}).get("low_confidence_intervals", [])

    per_slide = []
    for s in slide_result.get("segments", []):
        start = float(s.get("start_time", 0.0))
        end = float(s.get("end_time", start))
        speech_full = _text_for_interval(speech_segments, start, end, max_chars=5000)
        similarity = _text_similarity(s.get("ocr_text", ""), speech_full)
        speech_word_count = _text_word_count(speech_full)
        ocr_word_count = int(s.get("ocr_word_count", 0))
        speech_overlap_sec = 0.0
        for seg in speech_segments:
            seg_s = float(seg.get("start", 0.0))
            seg_e = float(seg.get("end", seg_s))
            speech_overlap_sec += _overlap_sec(seg_s, seg_e, start, end)
        speech_coverage_ratio = (speech_overlap_sec / max(1e-6, end - start))
        per_slide.append(
            {
                "slide_id": s.get("slide_id"),
                "start_time": start,
                "end_time": end,
                "duration": float(s.get("duration", max(0.0, end - start))),
                "ocr_text": s.get("ocr_text", ""),
                "description": s.get("description", ""),
                "layout": s.get("layout", ""),
                "image_path": s.get("image_path"),
                "speech": {
                    "wpm": _weighted_avg(rate_windows, "wpm", "start", "end", start, end),
                    "intelligibility": _weighted_avg(intel_segments, "score", "start", "end", start, end),
                    "filler_word_count": _count_occurrences(fillers_words, start, end),
                    "filler_phrase_count": _count_occurrences(fillers_phrases, start, end),
                    "noise_fraction": _noise_fraction(noise_intervals, start, end),
                    "speech_text": speech_full,
                    "speech_word_count": speech_word_count,
                    "speech_overlap_sec": round(float(speech_overlap_sec), 3),
                    "speech_coverage_ratio": round(float(speech_coverage_ratio), 3),
                    "speech_text_preview": speech_full[:300],
                    "evidence_intervals": {
                        "slow_speech": _clip_intervals(slow_intervals, start, end),
                        "fast_speech": _clip_intervals(fast_intervals, start, end),
                        "low_intelligibility": _clip_intervals(low_intel_intervals, start, end),
                        "background_noise": _clip_intervals(noise_intervals, start, end),
                    },
                    "filler_occurrences": {
                        "words": [f for f in fillers_words if start <= float(f.get("start", 0.0)) <= end][:10],
                        "phrases": [f for f in fillers_phrases if start <= float(f.get("start", 0.0)) <= end][:10],
                    },
                },
                "content_alignment": {
                    "similarity": similarity,
                    "ocr_word_count": ocr_word_count,
                },
            }
        )

    clothing_summary = None
    if clothing_result is not None:
        clothing_summary = {
            "summary": clothing_result.get("summary"),
            "description": clothing_result.get("description", ""),
            "style": clothing_result.get("style", ""),
            "top": clothing_result.get("top", ""),
            "is_appropriate": clothing_result.get("is_appropriate"),
            "recommendation": clothing_result.get("recommendation"),
            "detected_attributes": clothing_result.get("detected_attributes", []),
            "coverage": clothing_result.get("coverage", {}),
        }

    gaze_summary = None
    if gaze_result is not None:
        gaze_summary = {
            "overall_summary": gaze_result.get("overall_summary", {}),
            "issues": gaze_result.get("issues", {}),
            "meta": gaze_result.get("meta", {}),
        }

    emotion_summary = None
    if emotion_result is not None:
        emotion_summary = {
            "overall_stats": emotion_result.get("overall_stats", {}),
            "issues": emotion_result.get("issues", {}),
            "meta": emotion_result.get("meta", {}),
        }

    gesture_summary = None
    if gesture_result is not None:
        gesture_summary = {
            "overall": gesture_result.get("overall", {}),
            "issues": gesture_result.get("issues", {}),
            "meta": gesture_result.get("meta", {}),
        }

    overall = {
        "speech": {
            "filler_words": speech_result.get("filler_words", {}),
            "filler_phrases": speech_result.get("filler_phrases", {}),
            "intelligibility_global": speech_result.get("intelligibility", {}).get("global_score"),
            "noise_fraction": speech_result.get("background_noise", {}).get("fraction"),
            "avg_wpm": _weighted_avg(rate_windows, "wpm", "start", "end", 0.0, float("inf")),
        },
        "slides": {
            "count": len(slide_result.get("segments", [])),
            "avg_words": (
                sum([s.get("ocr_word_count", 0) for s in slide_result.get("segments", [])])
                / max(1, len(slide_result.get("segments", [])))
            ),
        },
        "clothing": clothing_summary,
        "gaze": gaze_summary,
        "emotion": emotion_summary,
        "gesture": gesture_summary,
    }

    sim_jaccards = [s.get("content_alignment", {}).get("similarity", {}).get("jaccard") for s in per_slide]
    sim_edits = [s.get("content_alignment", {}).get("similarity", {}).get("edit_ratio") for s in per_slide]
    avg_jaccard = sum([v for v in sim_jaccards if isinstance(v, (int, float))]) / max(1, len(sim_jaccards))
    avg_edit = sum([v for v in sim_edits if isinstance(v, (int, float))]) / max(1, len(sim_edits))

    multimodal_events = []
    g_slides = (gesture_result or {}).get("slide_summaries", {})
    gaze_slides = (gaze_result or {}).get("slide_summaries", {})
    emo_slides = (emotion_result or {}).get("slide_summaries", {})
    for sid, gs in g_slides.items():
        events = gs.get("events", [])
        gaze_ev = (gaze_slides.get(str(sid)) or {}).get("evidence_frames", [])
        emo_ev = (emo_slides.get(str(sid)) or {}).get("evidence_frames", [])
        for ev in events:
            st = float(ev.get("start_time", 0.0))
            en = float(ev.get("end_time", st))
            g_in = [g for g in gaze_ev if st <= float(g.get("t_sec", -1)) <= en]
            e_in = [e for e in emo_ev if st <= float(e.get("t_sec", -1)) <= en]
            multimodal_events.append(
                {
                    "slide_id": int(sid),
                    "event": ev.get("event"),
                    "start_time": st,
                    "end_time": en,
                    "gaze_labels": _label_counts(g_in, "gaze"),
                    "emotion_labels": _label_counts(e_in, "emotion"),
                }
            )

    content_report_payload = {
        "overall": {
            "speech": overall["speech"],
            "slides": overall["slides"],
        },
        "slides": per_slide,
        "derived_metrics": {
            "avg_slide_speech_similarity_jaccard": round(float(avg_jaccard), 3),
            "avg_slide_speech_similarity_edit": round(float(avg_edit), 3),
        },
    }

    visual_report_payload = {
        "overall": {
            "clothing": clothing_summary,
            "gaze": gaze_summary,
            "emotion": emotion_summary,
            "gesture": gesture_summary,
        },
        "events": {
            "gesture_by_slide": g_slides,
            "gaze_by_slide": gaze_slides,
            "emotion_by_slide": emo_slides,
            "multimodal_events": multimodal_events,
        },
    }

    return content_report_payload, visual_report_payload


def generate_report_pdf_from_video(video_path: str):
    if not video_path:
        raise ValueError("Video file is required.")

    log_lines: List[str] = []
    thresholds = load_thresholds(DEFAULT_THRESHOLDS_PATH)
    speech_cfg = thresholds["speech"]
    slides_cfg = thresholds["slides"]
    face_cache_cfg = thresholds["face_cache"]
    clothing_cfg = thresholds["clothing"]
    gaze_cfg = thresholds["gaze"]
    emotion_cfg = thresholds["emotion"]
    gesture_cfg = thresholds["gesture"]

    detectron_model_path = _resolve_path("DETECTRON_MODEL_PATH", DEFAULT_DETECTRON_MODEL_PATH)
    detectron_config_path = _resolve_path("DETECTRON_CONFIG_PATH", DEFAULT_DETECTRON_CONFIG_PATH)
    ocr_model_id = os.getenv("OCR_MODEL_ID", DEFAULT_OCR_MODEL_ID)
    whisper_model_size = os.getenv("WHISPER_MODEL_SIZE", "small")
    language = os.getenv("WHISPER_LANGUAGE", "en")

    logs_text = _push_log(log_lines, "Step 1/7: Speech analysis")
    yield logs_text, None
    speech_result = analyze_speech_tool(
        video_path=video_path,
        language=language,
        whisper_model_size=whisper_model_size,
        intelligibility_segment_len=speech_cfg["intelligibility_segment_len"],
    )

    logs_text = _push_log(log_lines, "Step 2/7: Slide detection and OCR")
    yield logs_text, None
    slide_result = slide_extraction_tool(
        video_path=video_path,
        model_path=detectron_model_path,
        config_path=detectron_config_path,
        sample_every_sec=slides_cfg["sample_every_sec"],
        ssim_thresh=slides_cfg["ssim_thresh"],
        min_segment_sec=slides_cfg["min_segment_sec"],
        similarity_threshold=slides_cfg["similarity_threshold"],
        min_word_count_for_slide=slides_cfg["min_word_count_for_slide"],
        ocr_sample_count=slides_cfg["ocr_sample_count"],
        top_k_ocr_frames=slides_cfg["top_k_ocr_frames"],
        sharpness_weight=slides_cfg["sharpness_weight"],
        ocr_model_id=ocr_model_id,
    )
    slide_count = len(slide_result.get("segments", []))
    logs_text = _push_log(log_lines, f"Slide analysis done with {slide_count} slides")
    yield logs_text, None

    logs_text = _push_log(log_lines, "Step 3/7: Face cache")
    yield logs_text, None
    face_cache_out = build_face_cache_tool(
        video_path=video_path,
        segments=slide_result.get("segments", []),
        per_slide_frames=face_cache_cfg["per_slide_frames"],
        batch_size=face_cache_cfg["face_batch_size"],
    )

    idx_to_slide = {
        int(s.get("slide_id")): {
            "slide_content": s.get("ocr_text", ""),
            "start_time": s.get("start_time"),
            "end_time": s.get("end_time"),
        }
        for s in slide_result.get("segments", [])
    }

    logs_text = _push_log(log_lines, "Step 4/7: Clothing analysis")
    yield logs_text, None
    clip_local_only = True if OFFLINE_MODE else clothing_cfg["clip_local_files_only"]
    clothing_classifier = ClothesCLIP(
        model_name=clothing_cfg["clip_model_name"],
        local_files_only=clip_local_only,
    )
    clothing_result = clothing_analysis_tool(
        video_path=video_path,
        slide_frame_mapping=face_cache_out.get("slide_frame_mapping", {}),
        face_crops_cache=face_cache_out.get("face_crops_cache", {}),
        clothing_classifier=clothing_classifier,
        frames_per_slide_max=clothing_cfg["clothing_frames_per_slide_max"],
        min_face_conf=clothing_cfg["clothing_min_face_conf"],
    )

    logs_text = _push_log(log_lines, "Step 4/7: Gaze analysis")
    yield logs_text, None
    gaze_heuristics = GazeHeuristics(
        pitch_down_thresh=gaze_cfg["pitch_down_thresh"],
        yaw_side_thresh=gaze_cfg["yaw_side_thresh"],
        pitch_center_thresh=gaze_cfg["pitch_center_thresh"],
        pitch_offset=gaze_cfg.get("pitch_offset", 0.0),
        yaw_center_thresh=gaze_cfg["yaw_center_thresh"],
    )
    gaze_estimator = MediaPipeGazeDirection(heuristics=gaze_heuristics)
    gaze_result = gaze_analysis_tool(
        video_path=video_path,
        slide_frame_mapping=face_cache_out.get("slide_frame_mapping", {}),
        face_crops_cache=face_cache_out.get("face_crops_cache", {}),
        gaze_estimator=gaze_estimator,
        idx_to_slide=idx_to_slide,
        frames_per_slide_max=gaze_cfg["frames_per_slide_max"],
        min_face_conf=gaze_cfg["min_face_conf"],
        min_gaze_conf=gaze_cfg["min_gaze_conf"],
        min_valid_frames_per_slide=gaze_cfg["min_valid_frames_per_slide"],
        min_coverage_ratio=gaze_cfg["min_coverage_ratio"],
        min_overall_valid_ratio=gaze_cfg["min_overall_valid_ratio"],
        expand_scale=gaze_cfg["expand_scale"],
        min_face_size=gaze_cfg["min_face_size"],
    )
    gaze_estimator.close()

    logs_text = _push_log(log_lines, "Step 4/7: Emotion analysis")
    yield logs_text, None
    emotion_device = "cuda" if torch.cuda.is_available() else "cpu"
    emotion_model_name = get_model_list()[0]
    fer = EmotiEffLibRecognizer(engine="onnx", model_name=emotion_model_name, device=emotion_device)
    emotion_result = emotion_analysis_tool(
        video_path=video_path,
        slide_frame_mapping=face_cache_out.get("slide_frame_mapping", {}),
        face_crops_cache=face_cache_out.get("face_crops_cache", {}),
        fer=fer,
        idx_to_slide=idx_to_slide,
        frames_per_slide_max=emotion_cfg["frames_per_slide_max"],
        min_face_conf=emotion_cfg["min_face_conf"],
        min_valid_frames_per_slide=emotion_cfg["min_valid_frames_per_slide"],
        expand_scale=emotion_cfg["expand_scale"],
        min_face_size=emotion_cfg["min_face_size"],
        batch_size=emotion_cfg["batch_size"],
    )
    del fer

    logs_text = _push_log(log_lines, "Step 4/7: Gesture analysis")
    yield logs_text, None
    gesture_heuristics = GestureHeuristics(
        hand_to_face_ratio=gesture_cfg["hand_to_face_ratio"],
        arms_crossed_wrist_ratio=gesture_cfg["arms_crossed_wrist_ratio"],
        arms_crossed_chest_y_ratio=gesture_cfg["arms_crossed_chest_y_ratio"],
        open_palms_ratio=gesture_cfg["open_palms_ratio"],
    )
    gesture_detector = Gestures(
        model_path=gesture_cfg["model_path"],
        conf=gesture_cfg["conf"],
        kp_conf=gesture_cfg["kp_conf"],
        heuristics=gesture_heuristics,
    )
    gesture_result = gesture_analysis_tool(
        video_path=video_path,
        slide_frame_mapping=face_cache_out.get("slide_frame_mapping", {}),
        gesture_detector=gesture_detector,
        idx_to_slide=idx_to_slide,
        frames_per_slide_max=gesture_cfg["frames_per_slide_max"],
        resize_w=gesture_cfg["resize_w"],
        max_total_frames=gesture_cfg["max_total_frames"],
        min_event_frames=gesture_cfg["min_event_frames"],
        min_pose_coverage_ratio=gesture_cfg["min_pose_coverage_ratio"],
        top_evidence_frames=gesture_cfg["top_evidence_frames"],
    )

    logs_text = _push_log(log_lines, "Step 5/7: Build payloads")
    yield logs_text, None
    content_payload, visual_payload = _build_payloads(
        speech_result=speech_result,
        slide_result=slide_result,
        clothing_result=clothing_result,
        gaze_result=gaze_result,
        emotion_result=emotion_result,
        gesture_result=gesture_result,
    )

    output_root = PROJECT_ROOT / "data" / "outputs" / "gradio"
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    logs_text = _push_log(log_lines, "Step 6/7: LLM recommendations and PDF")
    yield logs_text, None
    content_cfg = ContentRunConfig(max_concurrency=1, max_retries=4, backoff_base_sec=1.5)
    visual_cfg = VisualRunConfig(max_concurrency=1, max_retries=4, backoff_base_sec=1.5)
    results = asyncio.run(
        run_combined_recommendations(
            content_report_payload=content_payload,
            visual_report_payload=visual_payload,
            output_dir=run_dir,
            content_cfg=content_cfg,
            visual_cfg=visual_cfg,
        )
    )

    logs_text = _push_log(log_lines, "Step 7/7: Done")
    yield logs_text, results["paths"]["combined_pdf"]


def build_app() -> gr.Blocks:
    pdf_component = gr.PDF if hasattr(gr, "PDF") else gr.File

    with gr.Blocks(title="PRISM: Slide-Aligned Feedback for Presentation Skills") as demo:
        gr.HTML(
            """
            <section class="prism-hero">
              <h1 class="prism-title">PRISM: Slide-Aligned Feedback for Presentation Skills</h1>
              <p class="prism-subtitle">Upload a presentation video and receive a polished, slide-aware coaching report.</p>
              <div class="prism-badges">
                <span class="prism-badge">GPU-Ready</span>
                <span class="prism-badge">Slide + Speech Alignment</span>
                <span class="prism-badge">Visual Delivery Coaching</span>
              </div>
            </section>
            """
        )

        with gr.Row(elem_classes=["prism-grid"]):
            with gr.Column(scale=2, elem_classes=["prism-panel"]):
                gr.Markdown("### Upload your video")
                video_file = gr.File(
                    label="Presentation video",
                    file_types=[".mp4", ".mov", ".mkv", ".avi"],
                    type="filepath",
                    elem_classes=["prism-upload"],
                )
                run_btn = gr.Button("Generate PRISM Report", elem_classes=["prism-primary"])
            with gr.Column(scale=1, elem_classes=["prism-panel", "prism-side"]):
                gr.Markdown("### What happens next")
                gr.Markdown(
                    """
                    <div class="prism-step">1) Speech + slide sync to align talking points.</div>
                    <div class="prism-step">2) Gaze, gesture, emotion, clothing signals extracted.</div>
                    <div class="prism-step">3) Unified coaching PDF with slide-specific tips.</div>
                    """,
                )
                gr.Markdown("### Output")
                gr.Markdown("- One combined PDF report\n- Slide-by-slide evidence\n- Actionable coaching tips")

        with gr.Accordion("Live progress", open=False):
            log_out = gr.Textbox(label="Progress log", lines=12, interactive=False, elem_classes=["prism-log"])

        pdf_out = pdf_component(label="Combined PDF", elem_classes=["prism-pdf"])

        run_btn.click(
            fn=generate_report_pdf_from_video,
            inputs=[video_file],
            outputs=[log_out, pdf_out],
        )

    demo.queue()
    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(share=True, css=PAPER_GRADIENT_CSS)
