from __future__ import annotations

from pathlib import Path
from typing import Dict

from speech_analysis.audio_processing import SpeechProcessingSubsystem


def analyze_speech(
    video_path: str,
    language: str = "en",
    whisper_model_size: str = "small",
    intelligibility_segment_len: int = 10,
) -> Dict:
    video_path = str(Path(video_path).resolve())

    subsystem = SpeechProcessingSubsystem(
        video_path=video_path,
        lang=language,
        whisper_model_size=whisper_model_size,
        intelligibility_segment_len=intelligibility_segment_len,
    )
    return subsystem.run()


def analyze_speech_tool(
    video_path: str,
    language: str = "en",
    whisper_model_size: str = "small",
    intelligibility_segment_len: int = 10,
) -> Dict:
    """
    Backwards-compatible alias used by the tool registry / notebooks.
    """
    return analyze_speech(
        video_path=video_path,
        language=language,
        whisper_model_size=whisper_model_size,
        intelligibility_segment_len=intelligibility_segment_len,
    )
