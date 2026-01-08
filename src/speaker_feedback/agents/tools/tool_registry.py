# agents/tools/tool_registry.py
from __future__ import annotations

from typing import Any, Dict

from speaker_feedback.agents.tools.speech_analysis_tool import analyze_speech_tool as speech_analysis_tool
from speaker_feedback.agents.tools.slide_ocr_tool import detect_and_ocr_slides as slide_extraction_tool
from speaker_feedback.agents.tools.face_cache_tools import build_face_cache_tool
from speaker_feedback.agents.tools.clothing_tool import clothing_analysis_tool
from speaker_feedback.agents.tools.emotion_tool import emotion_analysis_tool
from speaker_feedback.agents.tools.gaze_tool import gaze_analysis_tool
from speaker_feedback.video_analysis.gesture_analysis import gestures_analysis_tool

from speaker_feedback.agents.tools.recommendation_tool import run_nemo_react_recommendations


def nemo_react_recommendation_tool(*, config_file: str, user_input: str) -> str:
    return run_nemo_react_recommendations(
        config_file=config_file,
        user_input=user_input,
    )
