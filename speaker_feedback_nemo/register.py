import json
from typing import Any, Dict

from pydantic import BaseModel, Field

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

from speaker_feedback_nemo.tools import load_payload, slim_payload, get_slide_context


class LoadPayloadInput(BaseModel):
    payload_path: str = Field(
        ...,
        description="Path to presentation_report.json or presentation_analysis.json.",
    )
    max_slide_text_chars: int = Field(
        default=800,
        description="Trim slide visual/spoken text to this length for the agent context.",
    )
    include_full_payload: bool = Field(
        default=True,
        description="Include trimmed payload in the tool response.",
    )


class GetSlideContextInput(BaseModel):
    payload_path: str = Field(
        ...,
        description="Path to presentation_report.json or presentation_analysis.json.",
    )
    slide_id: int = Field(
        ...,
        description="Slide ID to retrieve context for.",
    )
    max_slide_text_chars: int = Field(
        default=800,
        description="Trim slide visual/spoken text to this length.",
    )


class LoadPayloadConfig(FunctionBaseConfig, name="load_payload"):
    """Load and trim the presentation payload for the agent."""


class GetSlideContextConfig(FunctionBaseConfig, name="get_slide_context"):
    """Fetch a single slide context from a payload."""


@register_function(config_type=LoadPayloadConfig)
async def load_payload_tool(config: LoadPayloadConfig, builder: Builder):
    """Register tool for loading a presentation payload JSON."""

    async def _wrapper(
        payload_path: str,
        max_slide_text_chars: int = 800,
        include_full_payload: bool = True,
    ) -> str:
        payload = load_payload(payload_path)
        trimmed = slim_payload(payload, max_slide_text_chars=max_slide_text_chars)

        out: Dict[str, Any] = {
            "summary": trimmed.get("timeline_summary", {}),
            "meta": trimmed.get("meta", {}),
        }
        if include_full_payload:
            out["payload"] = trimmed

        return json.dumps(out, ensure_ascii=False)

    yield FunctionInfo.from_fn(
        _wrapper,
        input_schema=LoadPayloadInput,
        description=(
            "Load a presentation analysis JSON file and return a trimmed payload suitable "
            "for NeMo ReAct reasoning. Includes slide timeline, speech stats, and visual analyses."
        ),
    )


@register_function(config_type=GetSlideContextConfig)
async def get_slide_context_tool(config: GetSlideContextConfig, builder: Builder):
    """Register tool for retrieving a single slide context."""

    async def _wrapper(
        payload_path: str,
        slide_id: int,
        max_slide_text_chars: int = 800,
    ) -> str:
        payload = load_payload(payload_path)
        slide = get_slide_context(
            payload,
            slide_id,
            max_slide_text_chars=max_slide_text_chars,
        )
        return json.dumps(slide, ensure_ascii=False)

    yield FunctionInfo.from_fn(
        _wrapper,
        input_schema=GetSlideContextInput,
        description=(
            "Fetch a single slide's visual/spoken text, OCR tables/figures, and emotion/gaze/"
            "gesture context. Use this when you need details for a specific slide."
        ),
    )


def register() -> bool:
    """
    Entry point hook for NAT plugin discovery.
    The decorators above register tools at import time.
    """
    return True
