from __future__ import annotations

import json
from typing import Any, Dict, Optional


def _build_nemo_prompt(payload: Dict[str, Any]) -> str:
    return (
        "You are a presentation coach agent. Use the provided JSON payload to create "
        "actionable, per-slide and overall recommendations. Use a ReAct-style internal "
        "reasoning process, but do NOT output reasoning. Output only JSON with keys "
        "`overall` (list of strings) and `per_slide` (dict keyed by slide_id). "
        "Each per_slide entry should include `strengths` and `improvements`. "
        "Incorporate slide tables/figures from `ocr_parsed` when relevant. "
        "Keep each recommendation short and specific.\n\n"
        f"PAYLOAD:\n{json.dumps(payload, indent=2)}\n"
    )


def _run_nemo_agent(nemo_agent, prompt: str) -> str:
    if hasattr(nemo_agent, "run"):
        return nemo_agent.run(prompt)
    if hasattr(nemo_agent, "invoke"):
        return nemo_agent.invoke(prompt)
    if hasattr(nemo_agent, "chat"):
        return nemo_agent.chat(prompt)
    raise RuntimeError("Unsupported NeMo agent interface (no run/invoke/chat method).")


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 3:
            return parts[1].strip()
    return text


def _parse_nemo_json(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if raw is None:
        raise ValueError("NeMo response is empty.")
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="ignore")
    if not isinstance(raw, str):
        raw = str(raw)

    text = _strip_code_fence(raw)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start:end + 1])
    raise ValueError("NeMo response is not valid JSON.")


def build_recommendations(
    payload: Dict[str, Any],
    *,
    nemo_agent: Optional[object] = None,
) -> Dict[str, Any]:
    """
    Build recommendations using NeMo ReAct agent only.
    """
    if nemo_agent is None:
        raise ValueError("nemo_agent is required for recommendations.")

    prompt = _build_nemo_prompt(payload)
    raw = _run_nemo_agent(nemo_agent, prompt)
    parsed = _parse_nemo_json(raw)

    overall = parsed.get("overall", [])
    per_slide = parsed.get("per_slide", {})
    if not isinstance(overall, list) or not isinstance(per_slide, dict):
        raise ValueError("NeMo response must include `overall` list and `per_slide` dict.")

    raw_preview = raw if isinstance(raw, str) else json.dumps(raw)
    return {
        "overall": overall,
        "per_slide": per_slide,
        "meta": {
            "mode": "nemo",
            "raw_preview": raw_preview[:2000],
        },
    }
