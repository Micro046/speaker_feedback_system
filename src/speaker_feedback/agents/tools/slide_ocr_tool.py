from speaker_feedback.slide_analysis.slide_transition_ssim import detect_transitions_and_segments, build_predictor
from speaker_feedback.slide_analysis.slide_refine_ocr import SlideOCRRefiner, OCRRefineConfig
import gc
import torch 

from typing import Any, Dict

def detect_and_ocr_slides(
    video_path: str,
    model_path_detectron: str,
    config_path_detectron: str,
    ocr_model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    sample_every_sec: float = 1.0,
    ocr_sample_count: int = 3,
    top_k_ocr_frames: int = 2,
    sharpness_weight: float = 0.15,
) -> Dict[str, Any]:

    print("--- Step 1: Running Visual Analysis (SSIM) ---")
    ssim_results = detect_transitions_and_segments(
        video_path=video_path,
        model_path=model_path_detectron,
        config_path=config_path_detectron,
        sample_every_sec=sample_every_sec,
        ssim_thresh=0.85,
        min_segment_sec=10.0,
    )

    raw_segments = ssim_results["segments"]
    print(f"SSIM detected {len(raw_segments)} candidate segments.")

    print("--- Step 2: Running Semantic Analysis (OCR) ---")
    predictor = build_predictor(model_path_detectron, config_path_detectron, score_thresh=0.95)

    cfg = OCRRefineConfig(
        ocr_model_id=ocr_model_id,
        ocr_sample_count=ocr_sample_count,
        top_k_ocr_frames=top_k_ocr_frames,
        sharpness_weight=sharpness_weight,
    )
    refiner = SlideOCRRefiner(predictor=predictor, config=cfg)

    final_segments = refiner.refine_segments(video_path, raw_segments)
    refiner.unload()

    print(f"Refinement complete. Merged into {len(final_segments)} final slides.")

    return {
        "video_path": video_path,
        "raw_count": len(raw_segments),          
        "final_count": len(final_segments),    
        "raw_ssim_count": len(raw_segments),
        "final_slide_count": len(final_segments),
        "segments": final_segments,
        "full_text_content": "\n".join([f"[Slide {s['slide_id']}]: {s['ocr_text']}" for s in final_segments]),
    }


def slide_extraction_tool(
    video_path: str,
    model_path: str,
    config_path: str,
    *,
    sample_every_sec: float = 1.0,
    ssim_thresh: float = 0.82,
    min_segment_sec: float = 2.0,
    similarity_threshold: float = 0.78,
    min_word_count_for_slide: int = 15,
    ocr_model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    ocr_sample_count: int = 3,
    top_k_ocr_frames: int = 2,
    sharpness_weight: float = 0.15,
):
    ssim_res = detect_transitions_and_segments(
        video_path=video_path,
        model_path=model_path,
        config_path=config_path,
        sample_every_sec=sample_every_sec,
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
        ocr_sample_count=ocr_sample_count,
        top_k_ocr_frames=top_k_ocr_frames,
        sharpness_weight=sharpness_weight,
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
            "sample_every_sec": sample_every_sec,
            "ssim_thresh": ssim_thresh,
            "min_segment_sec": min_segment_sec,
            "similarity_threshold": similarity_threshold,
            "min_word_count_for_slide": min_word_count_for_slide,
            "ocr_model_id": ocr_model_id,
            "ocr_sample_count": ocr_sample_count,
            "top_k_ocr_frames": top_k_ocr_frames,
            "sharpness_weight": sharpness_weight,
        },
    }
