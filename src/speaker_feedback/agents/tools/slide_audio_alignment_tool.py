from typing import List, Dict, Any

def align_speech_with_slides(
    slides: List[Dict[str, Any]],
    speech_data: Dict[str, Any],
    margin: float = 0.5
) -> List[Dict[str, Any]]:
    """
    Aligns word-level speech transcripts with slide time windows.
    """
    # 1. Flatten words from speech_data
    words = []
    if "words" in speech_data and isinstance(speech_data["words"], list):
        words = speech_data["words"]
    elif "segments_raw" in speech_data:
        for seg in speech_data["segments_raw"]:
            if "words" in seg:
                words.extend(seg["words"])
    
    if not words:
        for slide in slides:
            slide["speech_transcript"] = ""
        return slides

    words.sort(key=lambda x: float(x.get("start", 0)))
    
    aligned_slides = []
    for slide in slides:
        s_start = float(slide.get("start_time", 0)) - margin
        s_end = float(slide.get("end_time", 0)) + margin
        
        slide_words = []
        for w in words:
            w_start = float(w.get("start", 0))
            w_end = float(w.get("end", 0))
            w_mid = (w_start + w_end) / 2
            
            if s_start <= w_mid <= s_end:
                slide_words.append(w.get("text", "").strip())
        
        transcript = " ".join(slide_words)
        
        new_slide = dict(slide)
        new_slide["speech_transcript"] = transcript
        aligned_slides.append(new_slide)
        
    return aligned_slides

def slide_audio_alignment_tool(
    slides: List[Dict[str, Any]],
    speech_output: Dict[str, Any]
) -> List[Dict[str, Any]]:
    return align_speech_with_slides(slides, speech_output)
