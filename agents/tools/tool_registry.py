from __future__ import annotations

import gc
import torch

from agents.tools.speech_analysis_tool import analyze_speech_tool
from agents.tools.face_cache_tools import build_face_cache_tool
from agents.tools.clothing_tool import clothing_analysis_tool
from agents.tools.emotion_tool import emotion_analysis_tool
from agents.tools.gaze_tool import gaze_analysis_tool
from agents.tools.gesture_tool import gestures_analysis_tool

from slide_analysis.slide_transition_ssim import detect_transitions_and_segments, build_predictor
from slide_analysis.slide_refine_ocr import SlideOCRRefiner, OCRRefineConfig


def speech_analysis_tool(video_path: str, **kwargs):
    return analyze_speech_tool(video_path, **kwargs)


def slide_extraction_tool(
    video_path: str,
    model_path: str,
    config_path: str,
    *,
    ssim_thresh: float = 0.82,
    min_segment_sec: float = 2.0,
    similarity_threshold: float = 0.78,
    min_word_count_for_slide: int = 15,
    ocr_model_id: str = "JackChew/Qwen2-VL-2B-OCR",
):
    ssim_res = detect_transitions_and_segments(
        video_path=video_path,
        model_path=model_path,
        config_path=config_path,
        ssim_thresh=ssim_thresh,
        min_segment_sec=min_segment_sec,
    )

    raw_segments = ssim_res["segments"]

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    predictor = build_predictor(model_path, config_path, score_thresh=0.95)
    refine_cfg = OCRRefineConfig(
        ocr_model_id=ocr_model_id,
        similarity_threshold=similarity_threshold,
        min_word_count_for_slide=min_word_count_for_slide,
    )
    refiner = SlideOCRRefiner(predictor=predictor, config=refine_cfg)

    try:
        final_segments = refiner.refine_segments(video_path, raw_segments)
    finally:
        refiner.unload()
        del refiner
        del predictor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "raw_count": len(raw_segments),
        "final_count": len(final_segments),
        "segments": final_segments,
        "ssim": ssim_res,
        "params": {
            "ssim_thresh": ssim_thresh,
            "min_segment_sec": min_segment_sec,
            "similarity_threshold": similarity_threshold,
            "min_word_count_for_slide": min_word_count_for_slide,
            "ocr_model_id": ocr_model_id,
        },
    }


def face_cache_tool(video_path, segments, fps):
    return build_face_cache_tool(video_path, segments, fps=fps)


def clothing_tool(video_path, slide_mapping, face_cache, classifier):
    return clothing_analysis_tool(video_path, slide_mapping, face_cache, classifier)


def emotion_tool(video_path, slide_mapping, face_cache, fer_engine, idx_to_slide):
    return emotion_analysis_tool(video_path, slide_mapping, face_cache, fer_engine, idx_to_slide)


def gaze_tool(video_path, slide_mapping, face_cache, gaze_estimator, idx_to_slide):
    return gaze_analysis_tool(video_path, slide_mapping, face_cache, gaze_estimator, idx_to_slide)


def gesture_tool(video_path, slide_mapping, gesture_detector, idx_to_slide):
    return gestures_analysis_tool(video_path, slide_mapping, gesture_detector, idx_to_slide)
