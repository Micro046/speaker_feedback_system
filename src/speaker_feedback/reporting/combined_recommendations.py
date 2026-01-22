from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents import Agent, OpenAIChatCompletionsModel, Runner, trace
from dotenv import load_dotenv
from openai import AsyncOpenAI
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    KeepTogether,
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

OPEN_ROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL_NAME = "nvidia/nemotron-3-nano-30b-a3b:free"


def _load_open_router_model(model_name: str = DEFAULT_MODEL_NAME) -> OpenAIChatCompletionsModel:
    load_dotenv(r"./.env", override=True)
    api_key = os.getenv("OPEN_ROUTER")
    if not api_key:
        raise RuntimeError("OPEN_ROUTER env var not set.")
    client = AsyncOpenAI(base_url=OPEN_ROUTER_BASE_URL, api_key=api_key)
    return OpenAIChatCompletionsModel(model=model_name, openai_client=client)


CONTENT_SLIDE_INSTRUCTION = """You are PresentationAlignmentCoach.

You will receive JSON with:
{
  "overall": {...},
  "derived_metrics": {...},
  "slide": {
     "slide_id": ...,
     "start_time": ...,
     "end_time": ...,
     "ocr_text": "...",
     "description": "...",
     "layout": "...",
     "content_alignment": {... optional ...},
     "speech": {
        "speech_text": "...",
        "wpm": ...,
        "intelligibility": ...,
        "noise_fraction": ...,
        "filler_word_count": ...,
        "filler_phrase_count": ...,
        "speech_coverage_ratio": ...
     }
  }
}

PRIMARY GOAL:
Evaluate speech-slide alignment using:
- FULL speech.speech_text (primary)
- FULL slide description (primary representation of slide meaning)
- OCR text (supporting signal: labels, proper nouns, numbers, exact terms)
- alignment metrics when available (jaccard/edit_ratio from slide.content_alignment or slide.speech.content_alignment)

DO NOT do keyword-matching as the main method. Use full meaning comparison and then summarize evidence.

EVIDENCE RULES (STRICT):
- Evidence must be 2-4 bullets.
- Evidence must explicitly compare:
  (a) what the speaker focuses on (from FULL speech_text),
  (b) what the slide shows (from FULL description, optionally OCR),
  (c) include at least one metric (jaccard or edit_ratio) in at least one bullet IF present.
- If partially_aligned or not_aligned, include at least one bullet describing the mismatch (focus/scope missing).

LAYOUT RULES (SECONDARY):
- Use layout only to briefly praise if clearly good OR warn if clearly harmful.
- Do NOT suggest color/contrast/diagram changes unless layout description explicitly indicates a problem.

Return ONLY valid JSON (no markdown) with this exact shape:

{
  "slide_id": <number or string>,
  "start_time": <number or null>,
  "end_time": <number or null>,
  "ocr_excerpt": "<short excerpt or key terms (not full OCR)>",
  "description_excerpt": "<short excerpt or key terms (not full description)>",
  "speech_excerpt": "<short excerpt or key terms (not full speech)>",
  "metrics": {
    "jaccard": <number or null>,
    "edit_ratio": <number or null>,
    "wpm": <number or null>,
    "intelligibility": <number or null>,
    "noise_fraction": <number or null>,
    "fillers": <number or null>,
    "speech_coverage_ratio": <number or null>
  },
  "alignment_assessment": "well_aligned" | "partially_aligned" | "not_aligned",
  "evidence": [ "<bullet>", "<bullet>" ],
  "layout_note": "<string or empty>",
  "recommendations": [ "<concrete action>", "<concrete action>", "... up to 5" ]
}

Constraints:
- Keep excerpts short (1-2 lines). Never paste full OCR or full speech.
- Do not invent numbers or fields. If missing, use null/empty.
"""


VISUAL_COACH_INSTRUCTION = """You are an Expert Presentation Coach specializing in non-verbal communication.

You will receive data for ONE SLIDE containing:
1. Gaze Distribution (where they looked: Slides vs Audience vs Script).
2. Facial Emotion Metrics (dominant emotion, confidence).
3. Gesture Analysis (open palms, crossed arms, pose coverage).
4. Context (slide ID, time).

YOUR GOAL:
Analyze the physical delivery for this specific segment.

INTERPRETATION RULES (CRITICAL):
1. "Anger" or "Disgust" Detection: In presentation contexts, CV models often misclassify
   concentration, squinting to read, or thinking as Anger/Disgust.
   - If gaze is on Slides or Script AND emotion is Anger/Disgust, interpret as
     intense concentration or straining to read. Advise them to relax their face.
   - Do NOT accuse the user of being angry unless it is overwhelmingly obvious.

2. Gaze:
   - High Slides percent (>60) => reading off the screen (weak connection).
   - High Script percent => reading notes (low confidence).
   - High Audience/Center percent => good.

3. Gestures:
   - Open Palms = good (inviting).
   - Arms Crossed = defensive/closed off.
   - Low Pose Coverage = bad camera framing (user is cut off).

OUTPUT FORMAT:
Return ONLY valid JSON:
{
  "slide_id": <id>,
  "status": "Needs Improvement" | "Good" | "Excellent",
  "summary": "<1 sentence summary of physical presence>",
  "gaze_feedback": "<Specific observation on eye contact>",
  "facial_feedback": "<Interpretation of expression>",
  "gesture_feedback": "<Observation on hands/posture>",
  "coaching_tip": "<One actionable command>"
}
"""


@dataclass
class ContentRunConfig:
    max_concurrency: int = 4
    max_retries: int = 2
    backoff_base_sec: float = 0.6


@dataclass
class VisualRunConfig:
    max_concurrency: int = 5
    max_retries: int = 4
    backoff_base_sec: float = 0.6


def _build_content_agent(model: OpenAIChatCompletionsModel) -> Agent:
    return Agent(
        name="content_alignment_agent",
        instructions=CONTENT_SLIDE_INSTRUCTION,
        model=model,
    )


def _build_visual_agent(model: OpenAIChatCompletionsModel) -> Agent:
    return Agent(
        name="visual_coach",
        instructions=VISUAL_COACH_INSTRUCTION,
        model=model,
    )


def compute_overall_statistics(presentation_json: dict) -> dict:
    overall = presentation_json.get("overall", {}) or {}
    speech = overall.get("speech", {}) or {}
    slides_meta = overall.get("slides", {}) or {}
    derived = presentation_json.get("derived_metrics", {}) or {}

    fillers_words = speech.get("filler_words", {}) or {}
    fillers_phrases = speech.get("filler_phrases", {}) or {}
    total_fillers = sum(fillers_words.values()) + sum(fillers_phrases.values())

    return {
        "slides_count": slides_meta.get("count"),
        "average_words_per_slide": slides_meta.get("avg_words"),
        "average_alignment_similarity_jaccard": derived.get("avg_slide_speech_similarity_jaccard"),
        "average_alignment_similarity_edit": derived.get("avg_slide_speech_similarity_edit"),
        "global_speech_wpm": speech.get("avg_wpm"),
        "global_intelligibility": speech.get("intelligibility_global"),
        "global_noise_fraction": speech.get("noise_fraction"),
        "total_fillers": total_fillers,
    }


def _extract_alignment_metrics(slide: dict) -> Dict[str, Any]:
    ca = slide.get("content_alignment", {}) or {}
    sca = (slide.get("speech", {}) or {}).get("content_alignment", {}) or {}

    def _get(cobj, key):
        sim = cobj.get("similarity", {}) or {}
        return sim.get(key) if isinstance(sim, dict) else None

    j = _get(ca, "jaccard")
    e = _get(ca, "edit_ratio")
    if j is None and e is None:
        j = _get(sca, "jaccard")
        e = _get(sca, "edit_ratio")
    return {"jaccard": j, "edit_ratio": e}


async def _run_agent_with_retries(
    *,
    agent: Agent,
    payload: dict,
    sem: asyncio.Semaphore,
    max_retries: int,
    backoff_base_sec: float,
    fallback_fn,
) -> dict:
    for attempt in range(1, max_retries + 2):
        try:
            async with sem:
                r = await Runner.run(agent, json.dumps(payload, ensure_ascii=False))
            return json.loads(r.final_output)
        except Exception as exc:
            if attempt >= (max_retries + 1):
                return fallback_fn(exc)
            await asyncio.sleep(backoff_base_sec * attempt)


async def _analyze_content_slide(
    slide: dict,
    overall: dict,
    derived: dict,
    agent: Agent,
    sem: asyncio.Semaphore,
    cfg: ContentRunConfig,
) -> dict:
    speech = slide.get("speech", {}) or {}
    sid = slide.get("slide_id")

    payload = {
        "overall": overall,
        "derived_metrics": derived,
        "slide": {
            "slide_id": slide.get("slide_id"),
            "start_time": slide.get("start_time"),
            "end_time": slide.get("end_time"),
            "ocr_text": slide.get("ocr_text"),
            "description": slide.get("description"),
            "layout": slide.get("layout"),
            "content_alignment": slide.get("content_alignment"),
            "speech": {
                "speech_text": speech.get("speech_text"),
                "wpm": speech.get("wpm"),
                "intelligibility": speech.get("intelligibility"),
                "noise_fraction": speech.get("noise_fraction"),
                "filler_word_count": speech.get("filler_word_count"),
                "filler_phrase_count": speech.get("filler_phrase_count"),
                "speech_coverage_ratio": speech.get("speech_coverage_ratio"),
            },
        },
    }

    def _fallback(exc: Exception) -> dict:
        align = _extract_alignment_metrics(slide)
        return {
            "slide_id": sid,
            "start_time": slide.get("start_time"),
            "end_time": slide.get("end_time"),
            "ocr_excerpt": "",
            "description_excerpt": "",
            "speech_excerpt": "",
            "metrics": {
                "jaccard": align.get("jaccard"),
                "edit_ratio": align.get("edit_ratio"),
                "wpm": speech.get("wpm"),
                "intelligibility": speech.get("intelligibility"),
                "noise_fraction": speech.get("noise_fraction"),
                "fillers": speech.get("filler_word_count"),
                "speech_coverage_ratio": speech.get("speech_coverage_ratio"),
            },
            "alignment_assessment": "not_aligned",
            "evidence": [
                f"Slide analysis failed for slide_id={sid}: {repr(exc)}",
            ],
            "layout_note": "",
            "recommendations": [
                "Retry later, reduce concurrency, or switch to a model that follows strict JSON more reliably.",
            ],
        }

    return await _run_agent_with_retries(
        agent=agent,
        payload=payload,
        sem=sem,
        max_retries=cfg.max_retries,
        backoff_base_sec=cfg.backoff_base_sec,
        fallback_fn=_fallback,
    )


async def generate_content_report(
    presentation_json: dict,
    *,
    agent: Optional[Agent] = None,
    cfg: Optional[ContentRunConfig] = None,
) -> dict:
    cfg = cfg or ContentRunConfig()
    if agent is None:
        agent = _build_content_agent(_load_open_router_model())

    overall = presentation_json.get("overall", {}) or {}
    derived = presentation_json.get("derived_metrics", {}) or {}
    slides = presentation_json.get("slides", []) or []

    sem = asyncio.Semaphore(cfg.max_concurrency)
    tasks = [
        _analyze_content_slide(s, overall, derived, agent, sem, cfg)
        for s in slides
    ]
    slide_results = await asyncio.gather(*tasks)

    return {
        "overall_statistics": compute_overall_statistics(presentation_json),
        "high_level_summary": [
            f"Processed {len(slide_results)} slides using full speech segments and slide meaning (description + OCR support).",
            "Use each slide's Evidence section to see what matched and what did not, backed by alignment metrics when available.",
        ],
        "slides": slide_results,
        "patterns": [],
        "action_plan": [],
    }


def _merge_visual_slide_data(visual_json: dict) -> List[dict]:
    events = visual_json.get("events", {}) or {}

    slide_ids = set()
    slide_ids.update((events.get("gaze_by_slide", {}) or {}).keys())
    slide_ids.update((events.get("emotion_by_slide", {}) or {}).keys())
    slide_ids.update((events.get("gesture_by_slide", {}) or {}).keys())

    merged_slides = []

    def _sort_key(x):
        try:
            return int(x)
        except Exception:
            return 9999

    for sid in sorted(list(slide_ids), key=_sort_key):
        gaze_data = (events.get("gaze_by_slide", {}) or {}).get(sid, {}) or {}
        emo_data = (events.get("emotion_by_slide", {}) or {}).get(sid, {}) or {}
        gest_data = (events.get("gesture_by_slide", {}) or {}).get(sid, {}) or {}

        merged_slides.append(
            {
                "slide_id": sid,
                "gaze": {
                    "distribution": gaze_data.get("focus_distribution", {}) or {},
                    "dominant": gaze_data.get("dominant_focus", "unknown"),
                },
                "emotion": {
                    "dominant": emo_data.get("dominant_emotion", "unknown"),
                    "distribution": emo_data.get("emotion_distribution", {}) or {},
                },
                "gesture": {
                    "open_palms_pct": (gest_data.get("joint_statistics", {}) or {}).get("open_palms_pct", 0),
                    "arms_crossed_pct": (gest_data.get("joint_statistics", {}) or {}).get("arms_crossed_pct", 0),
                    "pose_coverage": gest_data.get("pose_coverage", 0),
                },
            }
        )

    return merged_slides


def _visual_summary(visual_json: dict) -> dict:
    overall = visual_json.get("overall", {}) or {}

    clothing = overall.get("clothing", {}) or {}
    clothing_status = "Issue Detected" if not clothing.get("is_appropriate") else "Professional"
    clothing_rec = clothing.get("recommendation", "No specific recommendation.")
    clothing_desc = clothing.get("description", "N/A")

    gaze_ov = (overall.get("gaze", {}) or {}).get("overall_summary", {}) or {}
    focus_slides = (gaze_ov.get("focus_distribution", {}) or {}).get("slides", 0)
    focus_audience = (gaze_ov.get("focus_distribution", {}) or {}).get("audience", 0)

    emo_ov = (overall.get("emotion", {}) or {}).get("overall_stats", {}) or {}
    top_emo = emo_ov.get("most_common_emotion", "Neutral")
    if top_emo in {"Anger", "Disgust"}:
        top_emo = f"{top_emo} (Likely Concentration)"

    return {
        "clothing_status": clothing_status,
        "clothing_desc": clothing_desc,
        "clothing_rec": clothing_rec,
        "focus_slides": float(focus_slides),
        "focus_audience": float(focus_audience),
        "dominant_expression": top_emo,
    }


async def _analyze_visual_slide(
    slide_data: dict,
    agent: Agent,
    sem: asyncio.Semaphore,
    cfg: VisualRunConfig,
) -> dict:
    sid = slide_data.get("slide_id")

    def _fallback(exc: Exception) -> dict:
        return {
            "slide_id": sid,
            "status": "Analysis Failed",
            "summary": "Could not process visual metrics.",
            "gaze_feedback": "N/A",
            "facial_feedback": "N/A",
            "gesture_feedback": "N/A",
            "coaching_tip": f"Check raw data/output formatting. Last error: {repr(exc)}",
        }

    return await _run_agent_with_retries(
        agent=agent,
        payload=slide_data,
        sem=sem,
        max_retries=cfg.max_retries,
        backoff_base_sec=cfg.backoff_base_sec,
        fallback_fn=_fallback,
    )


async def generate_visual_report(
    visual_json: dict,
    *,
    agent: Optional[Agent] = None,
    cfg: Optional[VisualRunConfig] = None,
) -> dict:
    cfg = cfg or VisualRunConfig()
    if agent is None:
        agent = _build_visual_agent(_load_open_router_model())

    merged_data = _merge_visual_slide_data(visual_json)
    sem = asyncio.Semaphore(cfg.max_concurrency)
    tasks = [_analyze_visual_slide(s, agent, sem, cfg) for s in merged_data]
    slide_results = await asyncio.gather(*tasks)

    return {
        "summary": _visual_summary(visual_json),
        "slides": slide_results,
    }


def _safe(v: Any, default: str = "not provided") -> str:
    if v is None:
        return default
    if isinstance(v, float):
        return f"{v:.3f}".rstrip("0").rstrip(".")
    return str(v)


def _clean_text(text: str) -> str:
    if not text:
        return ""
    replacements = {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2022": "*",
        "\u2026": "...",
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    return text


def _fmt_paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(_clean_text(text), style)


def build_combined_pdf(content_report: dict, visual_report: dict, output_path: Path) -> Path:
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()

    color_primary = colors.HexColor("#2C3E50")
    color_accent = colors.HexColor("#16A085")
    color_bg_gray = colors.HexColor("#F2F4F6")
    color_ref_bg = colors.HexColor("#EAECEE")
    color_good = colors.HexColor("#27AE60")
    color_warn = colors.HexColor("#F39C12")
    color_bad = colors.HexColor("#C0392B")

    color_blue = colors.HexColor("#3498DB")
    color_orange = colors.HexColor("#E67E22")
    color_green = colors.HexColor("#27AE60")
    color_yellow = colors.HexColor("#F1C40F")

    title_style = ParagraphStyle(
        "MainTitle",
        parent=styles["Heading1"],
        fontSize=24,
        textColor=color_primary,
        alignment=TA_LEFT,
        spaceAfter=12,
    )
    h2_style = ParagraphStyle(
        "SectionHeader",
        parent=styles["Heading2"],
        fontSize=15,
        textColor=color_accent,
        spaceBefore=8,
        spaceAfter=6,
    )
    normal_style = ParagraphStyle(
        "CustomNormal",
        parent=styles["Normal"],
        fontSize=10,
        leading=13,
        alignment=TA_JUSTIFY,
        spaceAfter=4,
    )
    bullet_style = ParagraphStyle(
        "CustomBullet",
        parent=styles["Normal"],
        fontSize=10,
        leading=13,
        alignment=TA_JUSTIFY,
        leftIndent=14,
        bulletIndent=4,
        spaceAfter=2,
    )
    ref_key_style = ParagraphStyle(
        "RefKey",
        parent=styles["Normal"],
        fontSize=8,
        fontName="Helvetica-Bold",
        textColor=colors.dimgray,
    )
    ref_val_style = ParagraphStyle(
        "RefVal",
        parent=styles["Normal"],
        fontSize=8,
        fontName="Helvetica-Oblique",
        textColor=colors.black,
        leading=10,
        alignment=TA_JUSTIFY,
    )

    def _status_color(text: str) -> colors.Color:
        txt = (text or "").lower()
        if "well" in txt:
            return color_good
        if "partial" in txt:
            return color_warn
        if "not" in txt:
            return color_bad
        return colors.gray

    elements: List[Any] = []
    elements.append(_fmt_paragraph("Presentation Feedback Report", title_style))
    elements.append(Spacer(1, 6))

    # Content Alignment section
    elements.append(_fmt_paragraph("Content Alignment Report", h2_style))
    stats = content_report.get("overall_statistics", {}) or {}
    stat_data = [["Metric", "Value"]]
    stat_data.append(["Slides count", _safe(stats.get("slides_count"))])
    stat_data.append(["Average words per slide", _safe(stats.get("average_words_per_slide"))])
    stat_data.append(["Average alignment similarity (jaccard)", _safe(stats.get("average_alignment_similarity_jaccard"))])
    stat_data.append(["Average alignment similarity (edit)", _safe(stats.get("average_alignment_similarity_edit"))])
    stat_data.append(["Global speech WPM", _safe(stats.get("global_speech_wpm"))])
    stat_data.append(["Global intelligibility", _safe(stats.get("global_intelligibility"))])
    stat_data.append(["Global noise fraction", _safe(stats.get("global_noise_fraction"))])
    stat_data.append(["Total fillers", _safe(stats.get("total_fillers"))])

    stat_table = Table(stat_data, colWidths=[300, 150])
    stat_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), color_primary),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 10),
                ("BACKGROUND", (0, 1), (-1, -1), color_bg_gray),
                ("GRID", (0, 0), (-1, -1), 1, colors.white),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    elements.append(stat_table)
    elements.append(Spacer(1, 10))

    elements.append(_fmt_paragraph("Executive Summary", h2_style))
    for item in content_report.get("high_level_summary", []) or []:
        elements.append(_fmt_paragraph(f"- {item}", bullet_style))
    elements.append(Spacer(1, 8))

    elements.append(_fmt_paragraph("Detailed Slide Analysis", h2_style))
    for slide in content_report.get("slides", []) or []:
        header_elements: List[Any] = []
        status_color = _status_color(slide.get("alignment_assessment", ""))
        header_text = (
            f"<b>Slide {slide.get('slide_id', '?')}</b> "
            f"({slide.get('start_time', '?')}s-{slide.get('end_time', '?')}s) "
            f"| Status: {str(slide.get('alignment_assessment', 'unknown')).replace('_', ' ').upper()}"
        )
        header_para = Paragraph(header_text, ParagraphStyle("SlideHead", textColor=colors.white, fontSize=12, fontName="Helvetica-Bold"))
        head_table = Table([[header_para]], colWidths=[450])
        head_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), status_color),
                    ("PADDING", (0, 0), (-1, -1), 8),
                    ("ROUNDEDCORNERS", [5, 5, 5, 5]),
                ]
            )
        )
        header_elements.append(head_table)
        header_elements.append(Spacer(1, 4))

        ref_rows = []
        for key, val in [
            ("OCR excerpt", slide.get("ocr_excerpt", "")),
            ("Description excerpt", slide.get("description_excerpt", "")),
            ("Speech excerpt", slide.get("speech_excerpt", "")),
        ]:
            ref_rows.append([_fmt_paragraph(f"{key}:", ref_key_style), _fmt_paragraph(val or "not provided", ref_val_style)])

        ref_table = Table(ref_rows, colWidths=[80, 370])
        ref_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), color_ref_bg),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("PADDING", (0, 0), (-1, -1), 4),
                    ("BOX", (0, 0), (-1, -1), 0.5, colors.lightgrey),
                ]
            )
        )
        header_elements.append(ref_table)
        header_elements.append(Spacer(1, 6))

        metrics = slide.get("metrics", {}) or {}
        metric_text = "<br/>".join(
            [
                f"Alignment: jaccard={_safe(metrics.get('jaccard'))}, edit_ratio={_safe(metrics.get('edit_ratio'))}",
                "Speech: wpm={wpm}, intelligibility={intel}, noise_fraction={noise}, fillers={fillers}, coverage={cov}".format(
                    wpm=_safe(metrics.get("wpm")),
                    intel=_safe(metrics.get("intelligibility")),
                    noise=_safe(metrics.get("noise_fraction")),
                    fillers=_safe(metrics.get("fillers")),
                    cov=_safe(metrics.get("speech_coverage_ratio")),
                ),
            ]
        )
        p_metrics = Paragraph(f"<b>Key Metrics:</b><br/>{metric_text}", normal_style)
        m_table = Table([[p_metrics]], colWidths=[450])
        m_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FAFAFA")),
                    ("BOX", (0, 0), (-1, -1), 0.5, colors.silver),
                    ("PADDING", (0, 0), (-1, -1), 8),
                ]
            )
        )
        header_elements.append(m_table)
        header_elements.append(Spacer(1, 6))

        elements.append(KeepTogether(header_elements))

        evidence = slide.get("evidence", []) or []
        if evidence:
            elements.append(_fmt_paragraph("<b>Evidence & Analysis:</b>", normal_style))
            for item in evidence[:6]:
                elements.append(_fmt_paragraph(f"- {item}", bullet_style))
            elements.append(Spacer(1, 4))

        recs = slide.get("recommendations", []) or []
        if recs:
            rec_style = ParagraphStyle("RecTitle", parent=normal_style, textColor=color_primary, fontName="Helvetica-Bold")
            elements.append(_fmt_paragraph("Actionable Recommendations:", rec_style))
            for rec in recs[:6]:
                elements.append(_fmt_paragraph(f"- {rec}", bullet_style))

        elements.append(Spacer(1, 6))
        elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))
        elements.append(Spacer(1, 6))

    elements.append(Spacer(1, 8))

    # Visual Coaching section
    elements.append(_fmt_paragraph("Visual Coaching Report", ParagraphStyle("VisualTitle", parent=styles["Heading1"], fontSize=20, textColor=color_primary, spaceAfter=16)))
    summary = visual_report.get("summary", {}) or {}

    summary_rows = [
        [_fmt_paragraph("<b>Executive Summary</b>", ParagraphStyle("VisualH2", parent=styles["Normal"], fontSize=14, textColor=color_blue, fontName="Helvetica-Bold"))],
        [_fmt_paragraph(f"- Wardrobe Check: {summary.get('clothing_status', 'N/A')}", normal_style)],
        [_fmt_paragraph(f"  - Note: {summary.get('clothing_desc', 'N/A')}", normal_style)],
        [_fmt_paragraph(f"  - Advice: {summary.get('clothing_rec', 'N/A')}", normal_style)],
        [_fmt_paragraph(
            "- Primary Focus: {slides:.1f}% on Slides vs {aud:.1f}% on Audience.".format(
                slides=summary.get("focus_slides", 0.0),
                aud=summary.get("focus_audience", 0.0),
            ),
            normal_style,
        )],
        [_fmt_paragraph(f"- Dominant Expression: {summary.get('dominant_expression', 'N/A')}", normal_style)],
    ]
    t_summary = Table(summary_rows, colWidths=[450])
    t_summary.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), color_bg_gray),
                ("BOX", (0, 0), (-1, -1), 1, color_primary),
                ("PADDING", (0, 0), (-1, -1), 10),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    elements.append(t_summary)
    elements.append(Spacer(1, 10))

    elements.append(_fmt_paragraph("Detailed Delivery Analysis", ParagraphStyle("VisualSection", parent=styles["Heading2"], textColor=color_primary, spaceAfter=6)))
    for slide in visual_report.get("slides", []) or []:
        slide_elements = []

        status_clean = str(slide.get("status", "Unknown")).lower()
        if "good" in status_clean or "excellent" in status_clean:
            head_col = color_green
        elif "failed" in status_clean:
            head_col = colors.grey
        else:
            head_col = color_orange

        header_text = f"<b>Slide {slide.get('slide_id', '?')}</b> | Status: {slide.get('status', 'Unknown')}"
        t_head = Table([[Paragraph(header_text, ParagraphStyle("W", parent=normal_style, textColor=colors.white, fontSize=12, fontName="Helvetica-Bold"))]], colWidths=[450])
        t_head.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), head_col),
                    ("PADDING", (0, 0), (-1, -1), 8),
                    ("ROUNDEDCORNERS", [5, 5, 5, 5]),
                ]
            )
        )
        slide_elements.append(t_head)

        content_rows = [
            [_fmt_paragraph("<b>Observation:</b>", normal_style), _fmt_paragraph(slide.get("summary", "N/A"), normal_style)],
            [_fmt_paragraph("<b>Gaze:</b>", normal_style), _fmt_paragraph(slide.get("gaze_feedback", "N/A"), normal_style)],
            [_fmt_paragraph("<b>Face:</b>", normal_style), _fmt_paragraph(slide.get("facial_feedback", "N/A"), normal_style)],
            [_fmt_paragraph("<b>Body:</b>", normal_style), _fmt_paragraph(slide.get("gesture_feedback", "N/A"), normal_style)],
        ]

        tbl_style_cmds = [
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("PADDING", (0, 0), (-1, -1), 6),
            ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
        ]

        tip = slide.get("coaching_tip", "")
        if tip:
            tip_para = _fmt_paragraph(f"<b>COACH TIP:</b> {tip}", ParagraphStyle("Tip", parent=normal_style, textColor=color_primary))
            content_rows.append([tip_para, ""])
            last_row_idx = len(content_rows) - 1
            tbl_style_cmds.append(("SPAN", (0, last_row_idx), (-1, last_row_idx)))
            tbl_style_cmds.append(("BACKGROUND", (0, last_row_idx), (-1, last_row_idx), color_yellow))

        t_content = Table(content_rows, colWidths=[90, 360])
        t_content.setStyle(TableStyle(tbl_style_cmds))

        slide_elements.append(t_content)
        slide_elements.append(Spacer(1, 6))
        slide_elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))
        slide_elements.append(Spacer(1, 6))
        elements.append(KeepTogether(slide_elements))

    doc.build(elements)
    return output_path


async def run_combined_recommendations(
    *,
    content_report_payload: dict,
    visual_report_payload: dict,
    output_dir: Path,
    model_name: str = DEFAULT_MODEL_NAME,
    content_cfg: Optional[ContentRunConfig] = None,
    visual_cfg: Optional[VisualRunConfig] = None,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = _load_open_router_model(model_name)
    content_agent = _build_content_agent(model)
    visual_agent = _build_visual_agent(model)

    content_task = generate_content_report(content_report_payload, agent=content_agent, cfg=content_cfg)
    visual_task = generate_visual_report(visual_report_payload, agent=visual_agent, cfg=visual_cfg)

    with trace("combined_recommendations"):
        content_report, visual_report = await asyncio.gather(content_task, visual_task)

    pdf_path = output_dir / "presentation_feedback_report.pdf"
    build_combined_pdf(content_report, visual_report, pdf_path)

    return {
        "content_report": content_report,
        "visual_report": visual_report,
        "paths": {
            "combined_pdf": str(pdf_path),
        },
    }
