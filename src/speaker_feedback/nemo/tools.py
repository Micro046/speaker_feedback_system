# speaker_feedback/nemo/tools.py
from __future__ import annotations

import json
from typing import Any, Dict


def build_react_user_input(
    *,
    payload: Dict[str, Any],
    constraints: Dict[str, str] | None = None,
    top_k_recommendations: int = 6,
) -> str:
    """
    Builds a strict ReAct-friendly instruction + embeds payload JSON.

    payload: your full computed analysis dict (speech/slides/clothing/emotion/gaze/gesture...)
    constraints: presentation type, audience, goal, time limit, etc.
    """
    constraints = constraints or {}
    payload_json = json.dumps(payload, ensure_ascii=False)

    # Keep instructions very explicit so the agent produces defensible recs.
    return f"""
You are a presentation coach. Produce recommendations ONLY if you can support them with evidence.
If a field is missing from PAYLOAD_JSON, say it is not available and do not guess.

CONSTRAINTS (must follow):
- Presentation type: {constraints.get("presentation_type","")}
- Audience: {constraints.get("audience","")}
- Goal: {constraints.get("goal","")}
- Time limit: {constraints.get("time_limit","")}

OUTPUT FORMAT (strict):
Return a Markdown report. Use these sections in order:
1) "# Presentation Feedback Report"
2) "## Overall Stats" (bullet list with numeric facts and derived_metrics)
3) "## Presentation Context" (presentation type + inferred role/designation + delivery mode)
4) "## Strengths" (bullet list, evidence-backed)
5) "## Recommendations" (exactly {top_k_recommendations} items, numbered)
6) "## Slide-by-Slide" (one sub-section per slide_id)

Each recommendation item must include:
- Title (short)
- What to change (1-2 sentences)
- Evidence (must include slide_id, time range, and measured facts)
- Derived metrics (state formula or definition)
- Confidence (0-1) with a reason using coverage/confidence fields
- A 30-60 second drill (actionable)

Each slide section must include:
- slide_id and time range
- slide content excerpt (from slide_summaries or segments)
- spoken excerpt (from slide_summaries or segments)
- Alignment check: are spoken keywords aligned with slide content?
  Use content_overlap_jaccard and visual/spoken keyword coverage.
- Strengths (1-2 bullets, evidence-backed)
- Improvements (1-3 bullets, evidence-backed)
- NOTE: Per-slide feedback must ONLY use speech + slide content (no gaze/gesture/clothing).

RULES:
- If a modality has low reliability (example: gaze valid_gaze_ratio < 0.3, pose coverage < 0.4),
  you MUST downweight it and say confidence is lower.
- If emotion coverage or confidence is low (coverage_ratio < 0.4 or avg_confidence < 0.4), downweight it.
- If clothing frames_used is small (< 6), downweight clothing evidence.
- Clothing override: if clothing style/top/description mentions "blazer" or "suit",
  treat the attire as appropriate and ignore color-tone warnings.
- Clothing is overall only. Do NOT assign clothing feedback per slide.
- If policy.presentation_mode is "online", do not use gaze/gesture/clothing in recommendations
  unless confidence is high and the user explicitly requested visual coaching.
- Per-slide feedback must focus on slide content + spoken content alignment and clarity.
- Prefer recommendations that improve judge scoring: clarity, structure, persuasion.
- Tone guardrail: keep feedback constructive and professional; never insulting.
- Safety: never invent timestamps, metrics, or evidence not present in PAYLOAD_JSON.
- Do not introduce numeric targets/thresholds/benchmarks unless they are in PAYLOAD_JSON or constraints.
- Encourage strengths when evidence supports them.
- Infer presentation type (student/academic/industry) and presenter role (student/CEO/engineer/manager)
  from presentation_context, constraints, or content cues; if unknown, say "unknown" and avoid role-specific advice.

TASK (ReAct steps required):
- Step A: Select the 2-4 biggest issues by impact (clarity + persuasion).
- Step B: For each issue, pull the best evidence snippets (timestamps + slide_id).
- Step C: Use derived_metrics if provided; otherwise compute only from available fields:
  - speech: coverage ratios, word counts, or wpm only if present
  - alignment: content_overlap_jaccard, visual_keyword_coverage, spoken_keyword_coverage
- Step D: Write the final recommendations.

PAYLOAD_JSON:
{payload_json}
""".strip()


def build_visual_coaching_input(
    *,
    payload: Dict[str, Any],
    constraints: Dict[str, str] | None = None,
    top_k_recommendations: int = 6,
) -> str:
    constraints = constraints or {}
    payload_json = json.dumps(payload, ensure_ascii=False)

    return f"""
You are a presentation coach. Produce a Visual Coaching report ONLY using the provided payload.
If a field is missing, say it is not available and do not guess.
Favor evidence-backed strengths, not just weaknesses.

CONSTRAINTS (must follow):
- Presentation type: {constraints.get("presentation_type","")}
- Audience: {constraints.get("audience","")}
- Goal: {constraints.get("goal","")}
- Time limit: {constraints.get("time_limit","")}

OUTPUT FORMAT (strict):
Return a Markdown report. Use these sections in order:
1) "# Visual Coaching Report"
2) "## Overall Visual Stats" (bullet list with numeric facts + data_quality)
3) "## Strengths" (bullet list, evidence-backed)
4) "## Weaknesses" (bullet list, evidence-backed)
5) "## Recommendations" (exactly {top_k_recommendations} items, numbered)
6) "## Timestamped Evidence" (bullet list of key intervals with modality cues)

Each recommendation item must include:
- Title (short)
- What to change (1-2 sentences)
- Evidence (must include timestamp intervals and measured facts)
- Confidence (0-1) with a reason using coverage/confidence fields
- A 30-60 second drill (actionable)

VISUAL LOGIC (use these patterns when evidence exists):
- Defensiveness: Arms Crossed + Angry/Neutral emotion around same timestamp.
- Nervousness: Looking down + Hands in Pockets + high filler count (if present in payload).
- Engagement: Open Palms + Audience gaze + Happy/Neutral.
- Dress Code Audit: CLIP summary indicates casual attire but context is formal.
Use payload.events.multimodal_events when available to cite combined signals at exact timestamps.

RULES:
- Use gaze/emotion/gesture/clothing ONLY (no slide content).
- If data_quality is low for a modality, downweight it and say confidence is lower.
- Clothing override: if clothing style/top/description mentions "blazer" or "suit",
  treat the attire as appropriate and ignore color-tone warnings.
- Tone guardrail: keep feedback constructive and professional; never insulting.
- Safety: never invent timestamps, metrics, or evidence not present in PAYLOAD_JSON.
- Do not introduce numeric targets/thresholds/benchmarks unless they are in PAYLOAD_JSON or constraints.

PAYLOAD_JSON:
{payload_json}
""".strip()


def build_storytelling_input(
    *,
    payload: Dict[str, Any],
    constraints: Dict[str, str] | None = None,
    top_k_recommendations: int = 6,
) -> str:
    constraints = constraints or {}
    payload_json = json.dumps(payload, ensure_ascii=False)

    return f"""
You are a presentation coach. Produce a Slide + Speech (Storytelling) report ONLY using the provided payload.
If a field is missing, say it is not available and do not guess.
Favor evidence-backed strengths, not just weaknesses.

CONSTRAINTS (must follow):
- Presentation type: {constraints.get("presentation_type","")}
- Audience: {constraints.get("audience","")}
- Goal: {constraints.get("goal","")}
- Time limit: {constraints.get("time_limit","")}

OUTPUT FORMAT (strict):
Return a Markdown report. Use these sections in order:
1) "# Slide + Speech Feedback Report"
2) "## Overall Story Stats" (bullet list with numeric facts + derived_metrics)
3) "## Strengths" (bullet list, evidence-backed)
4) "## Weaknesses" (bullet list, evidence-backed)
5) "## Recommendations" (exactly {top_k_recommendations} items, numbered)
6) "## Slide-by-Slide" (one sub-section per slide_id)

Each recommendation item must include:
- Title (short)
- What to change (1-2 sentences)
- Evidence (must include slide_id, time range, and measured facts)
- Derived metrics (state formula or definition)
- Confidence (0-1) with a reason using coverage/confidence fields
- A 30-60 second drill (actionable)

Each slide section must include:
- slide_id and time range
- slide content excerpt (from slides, slide_summaries, or segments)
- spoken excerpt (from slides, slide_summaries, or segments)
- Alignment check: are spoken keywords aligned with slide content?
  Use content_alignment.similarity.jaccard or content_overlap_jaccard when available.
- Strengths (1-2 bullets, evidence-backed)
- Weaknesses (1-3 bullets, evidence-backed)

Content-specific triggers (use when evidence exists):
- Reader Trap: high similarity between slide text and spoken text.
- Visual Overload: OCR word count high + fast WPM in the same slide.
- Visual Disconnect: low keyword overlap between slide and speech.
- Filler-Content Ratio: high fillers on complex slides.

RULES:
- Use speech + slide content ONLY. Do NOT use gaze/gesture/clothing/emotion.
- Use speech evidence intervals (fast/slow speech, low intelligibility, noise) when available.
- Tone guardrail: keep feedback constructive and professional; never insulting.
- Safety: never invent timestamps, metrics, or evidence not present in PAYLOAD_JSON.
- Do not introduce numeric targets/thresholds/benchmarks unless they are in PAYLOAD_JSON or constraints.
- If payload has "slides", use:
  - slides[*].ocr_text or slides[*].description for slide content
  - slides[*].speech.speech_text_preview for spoken excerpt
  - slides[*].content_alignment.similarity.jaccard for overlap
  - slides[*].speech.evidence_intervals for time ranges
- If derived_metrics uses avg_slide_speech_similarity_jaccard or avg_slide_speech_similarity_edit,
  cite those instead of missing content_overlap_jaccard fields.

PAYLOAD_JSON:
{payload_json}
""".strip()
