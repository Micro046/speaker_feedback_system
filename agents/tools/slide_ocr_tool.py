from __future__ import annotations
from typing import Dict, Any, Optional

from slide_analysis.slide_transition_ssim import detect_transitions_and_segments
from slide_analysis.slide_refine_ocr import SlideOCRRefiner

def detect_and_ocr_slides(
    video_path: str,
    model_path_detectron: str, # For bounding box/SSIM
    config_path_detectron: str,
    ocr_model_id: str = "JackChew/Qwen2-VL-2B-OCR"
) -> Dict[str, Any]:
    """
    Full pipeline: 
    1. Detects candidate segments using SSIM (visual).
    2. Runs Qwen2-VL OCR on candidates.
    3. Merges segments with similar text (removes animation artifacts).
    """
    
    # Step 1: Fast SSIM Detection
    print("--- Step 1: Running Visual Analysis (SSIM) ---")
    ssim_results = detect_transitions_and_segments(
        video_path=video_path,
        model_path=model_path_detectron,
        config_path=config_path_detectron,
        # Slightly looser thresholds because OCR will clean up false positives
        ssim_thresh=0.85, 
        min_segment_sec=1.0 
    )
    
    raw_segments = ssim_results["segments"]
    print(f"SSIM detected {len(raw_segments)} candidate segments.")

    # Step 2: Semantic OCR Refinement
    print("--- Step 2: Running Semantic Analysis (OCR) ---")
    refiner = SlideOCRRefiner(model_id=ocr_model_id)
    
    final_segments = refiner.refine_segments(video_path, raw_segments)
    print(f"Refinement complete. Merged into {len(final_segments)} final slides.")

    return {
        "video_path": video_path,
        "raw_ssim_count": len(raw_segments),
        "final_slide_count": len(final_segments),
        "segments": final_segments,
        "full_text_content": "\n".join([f"[Slide {s['slide_id']}]: {s['ocr_text']}" for s in final_segments])
    }