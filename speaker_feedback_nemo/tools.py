# speaker_feedback_nemo/tools.py
from __future__ import annotations

import json
from typing import Any, Dict


def build_react_user_input(
    *,
    payload: Dict[str, Any],
    constraints: Dict[str, str],
    top_k_recommendations: int = 6,
) -> str:
    """
    Builds a strict ReAct-friendly instruction + embeds payload JSON.

    payload: your full computed analysis dict (speech/slides/clothing/emotion/gaze/gesture...)
    constraints: presentation type, audience, goal, time limit, etc.
    """
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
3) "## Strengths" (bullet list, evidence-backed)
4) "## Recommendations" (exactly {top_k_recommendations} items, numbered)
5) "## Slide-by-Slide" (one sub-section per slide_id)

Each recommendation item must include:
- Title (short)
- What to change (1-2 sentences)
- Evidence (must include slide_id, time range, and measured facts)
- Derived metrics (state formula or definition)
- Confidence (0-1) with a reason using coverage/confidence fields
- A 30-60 second drill (actionable)

Each slide section must include:
- slide_id and time range
- slide content excerpt (from visual_text or slide_summaries)
- spoken excerpt (from spoken_text or slide_summaries)
- Strengths (1-2 bullets, evidence-backed)
- Improvements (1-3 bullets, evidence-backed)

RULES:
- If a modality has low reliability (example: gaze valid_gaze_ratio < 0.3, pose coverage < 0.4),
  you MUST downweight it and say confidence is lower.
- If emotion coverage or confidence is low (coverage_ratio < 0.4 or avg_confidence < 0.4), downweight it.
- If clothing frames_used is small (< 6), downweight clothing evidence.
- Clothing is overall only. Do NOT assign clothing feedback per slide.
- Prefer recommendations that improve judge scoring: clarity, structure, persuasion.
- Never invent timestamps or metrics not present in payload.
- Encourage strengths when evidence supports them.

TASK (ReAct steps required):
- Step A: Select the 2-4 biggest issues by impact (clarity + persuasion).
- Step B: For each issue, pull the best evidence snippets (timestamps + slide_id).
- Step C: Use derived_metrics if provided; otherwise compute only from available fields:
  - speech: coverage ratios, word counts, or wpm only if present
  - gaze: left/right/center distribution, valid ratio
  - gesture: pose coverage or joint movement stats
- Step D: Write the final recommendations.

PAYLOAD_JSON:
{payload_json}
""".strip()
