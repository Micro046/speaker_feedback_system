# Speaker Feedback System Workflow (Step-by-Step)

This document describes the complete end-to-end workflow, including all thresholds and defaults. It matches the current notebook (`speaker_feedback_system.ipynb`) and supporting modules.

## 0) Inputs and Outputs

Inputs:
- Video file: `video/VK_New_Video.mp4`
- Slide detector weights: `model_cache/weights/model_best.pth`
- Slide detector config: `model_cache/weights/my_custom_config.yaml`
- Optional: `NVIDIA_API_KEY` for NeMo Agent Toolkit (NAT) ReAct runs

Primary outputs:
- `outputs/presentation_analysis.json` (baseline analysis payload)
- `outputs/analysis_payload_for_nemo.json` (full payload for NeMo agent)
- `outputs/nemo_recommendations.json` (NeMo agent output)
- `outputs/presentation_report.md` (markdown report)
- `outputs/presentation_report.json` (final report payload + recommendations)

## 1) Environment and Cache Setup

Files and paths:
- Model cache directory is `model_cache`.
- `XDG_CACHE_HOME` is set to `model_cache`.
- `imageio_ffmpeg.get_ffmpeg_exe()` is added to PATH for Whisper.

Notebook step: "Model Cache and Paths".

## 2) Speech Analysis (ASR + Metrics)

Tool:
- `agents/tools/speech_analysis_tool.py`
- Under the hood: `speech_analysis/audio_processing.py`

What happens:
1. Extract audio from video via `moviepy.VideoFileClip`.
2. Run `whisper_timestamped.transcribe` with `detect_disfluencies=True`.
3. Build word/segment lists and compute metrics.

Thresholds and settings:
- Whisper model: `small` (default in `speech_analysis_tool`)
- Intelligibility segment length: 30 seconds (notebook override)
- Speech rate window: 60 seconds (`SpeechRate.WINDOW`)
- Speech rate thresholds: slow < 60 words/min, fast > 140 words/min (`SpeechRate.MIN/MAX`)
- Background noise window: 30 seconds (`BackgroundNoise.window`)
- Background noise ratio threshold: 0.45 (`BackgroundNoise.threshold`)
- Intelligibility: STOI per segment length (`Intelligibility.segment_len`)
- Filler word/phrase lists are in `FillerWordsAndPhrases.EN`

Output:
- `speech_out` includes `transcription`, `segments`, `filler_words`, `filler_phrases`, `speech_rate`, `background_noise`, `intelligibility`

Notebook step: "Speech Analysis".

## 3) Slide Detection (SSIM + OCR Refinement)

Tool:
- `agents/tools/tool_registry.py` -> `slide_extraction_tool`
- Visual detection: `slide_analysis/slide_transition_ssim.py`
- OCR refinement: `slide_analysis/slide_refine_ocr.py`

What happens:
1. Detect slide bounding boxes with Detectron2 (`build_predictor`, score threshold 0.95).
2. Sample frames and compute SSIM between consecutive slide crops.
3. Segment video into candidate slide windows.
4. OCR each candidate window using Qwen2-VL and merge duplicates by text similarity.

SSIM thresholds and settings (defaults unless overridden in notebook):
- `sample_every_sec`: 0.75 sec
- `downscale_max_side`: 960
- `score_thresh`: 0.95 (Detectron2 ROI head)
- `ssim_thresh`: 0.70 (notebook override; default is 0.82 in code)
- `debounce_sec`: 1.0
- `min_segment_sec`: 10.0 (notebook override; default is 2.0 in code)

OCR refinement settings (`slide_analysis/slide_refine_ocr.py`):
- OCR model: `JackChew/Qwen2-VL-2B-OCR`
- `similarity_threshold`: 0.78
- `min_token_count_for_similarity`: 5
- `edit_prefix_chars`: 200
- `w_jaccard`: 0.6
- `w_edit`: 0.4
- `min_word_count_for_slide`: 15
- Multi-sample OCR times: 10%, 50%, 90% of slide duration
- OCR prompt: "Extract all clearly readable text from this slide. Return plain text only. Do not repeat content. Do not add explanations."

Output:
- `slides_out` with `raw_count`, `final_count`, `segments`, `ssim` params
- Each segment contains `slide_id`, `start_time`, `end_time`, `duration`, and OCR fields

Notebook step: "Slide Detection (SSIM + OCR)".

## 4) Align Speech with Slides

Process:
- For each slide window, collect speech segments whose timestamps overlap.
- Build `final_timeline` with:
  - `slide_id`, `start_time`, `end_time`, `duration`
  - `visual_text` (OCR)
  - `visual_word_count`
  - `spoken_text`

Output:
- `final_timeline` list and `idx_to_slide` mapping

Notebook step: "Align Audio with Slides".

## 5) Parse OCR Content (Tables, Figures, Bullets)

Tool:
- `slide_analysis/slide_content_parser.py`

What happens:
- Extract bullets, tables, figure captions, and clean text from OCR.
- Tables are detected by `|` separators or 2+ spaces / tab-delimited columns.
- Figures and tables are detected by regex patterns:
  - Figures: `figure`, `fig.`
  - Tables: `table`, `tbl.`

Output:
- `ocr_parsed` is attached to each slide in `final_timeline`.

Notebook step: "Parse OCR Content (Tables and Figures)".

## 6) Persist Baseline JSON

Output file:
- `outputs/presentation_analysis.json`

Includes:
- `meta`, `speech_stats`, `slides_debug`, `timeline`

Notebook step: "Persist Baseline JSON".

## 7) Face Cache (Shared Visual Hub)

Tool:
- `agents/tools/face_cache_tools.py` -> `build_face_cache_tool`
- Core logic: `video_analysis/frame_sampling_face_cache.py`

What happens:
- Sample frames per slide (uniform sampling with edge padding).
- Run MTCNN to detect the best face per frame.
- Cache face bounding boxes and confidence scores.

Sampling thresholds (defaults):
- `per_slide_frames`: 12
- `edge_pad_sec`: 0.2
- `edge_pad_ratio`: 0.05

Face detection thresholds (defaults in `FaceCacheConfig`):
- `resize_max_width`: 640
- `min_face_size`: 20
- `prob_thresh`: 0.5
- `mtcnn_thresholds`: (0.3, 0.4, 0.5)
- `mtcnn_factor`: 0.7
- `batch_size`: 24 (notebook sets same)

Output:
- `slide_frame_mapping`, `face_crops_cache`, face stats

Notebook step: "Face Cache (Shared for Visual Analyses)".

## 8) Clothing Analysis (Overall)

Tool:
- `agents/tools/clothing_tool.py` -> `clothing_analysis_tool`
- Core logic: `video_analysis/clothing_analysis.py`
- Classifier: `video_analysis/clothing_model.py` (CLIP)

Thresholds (defaults):
- `frames_per_slide_max`: 4
- `min_face_conf`: 0.55
- Torso crop scaling: `torso_scale_w=2.2`, `torso_scale_h=3.2`, `torso_shift_y=1.15`
- `max_total_frames`: 300
- `read_every_nth`: 1

Output:
- `clothing_analysis` summary with overall recommendation and coverage

Notebook step: "Clothing Analysis (Overall)".

## 9) Emotion Analysis (Per-slide + Overall)

Tool:
- `agents/tools/emotion_tool.py` -> `emotion_analysis_tool`
- Core logic: `video_analysis/emotion_analysis.py`
- Model: `EmotiEffLibRecognizer` (engine `onnx`)

Thresholds (defaults):
- `frames_per_slide_max`: 6
- `min_face_conf`: 0.55
- `expand_scale`: 1.25
- `min_face_size`: 48
- `batch_size`: 32
- `min_valid_frames_for_slide`: 1
- `max_total_frames`: 400

Output:
- Overall emotion stats and per-slide summaries
- Slide timeline is updated with `dominant_emotion` and `emotion_confidence`

Notebook step: "Emotion Analysis".

## 10) Gaze Analysis (Overall + Per-slide)

Tool:
- `agents/tools/gaze_tool.py` -> `gaze_analysis_tool`
- Core logic: `video_analysis/gaze_analysis.py`
- Estimator: `video_analysis/gaze_estimator.py` (MediaPipe FaceMesh)

Thresholds (gaze analysis):
- `frames_per_slide_max`: 6
- `min_face_conf`: 0.55
- `expand_scale`: 1.3
- `min_face_size`: 48
- `max_total_frames`: 300
- Slide issue rules:
  - `valid_gaze_ratio < 0.3` -> "Low confidence eye contact"
  - `left > 40%` or `right > 40%` -> excessive gaze direction

Thresholds (gaze estimator):
- `center_thresh`: 0.02
- `max_delta`: 0.06

Output:
- Overall gaze summary and per-slide summaries

Notebook step: "Gaze Analysis (Overall)".

## 11) Gesture Analysis (Overall + Per-slide)

Tool:
- `agents/tools/gesture_tool.py` -> `gestures_analysis_tool`
- Core model: `video_analysis/gesture_analysis.py` (YOLOv8 Pose)

Thresholds:
- YOLO pose conf: 0.3 (model init)
- Joint keypoint conf: >= 0.30 for each joint in a triplet
- Movement score heuristics:
  - `score == 0` -> "No reliable pose data"
  - `score > 18` -> high movement
  - `score < 6` -> low movement

Notebook settings:
- `frames_per_slide_max`: 6

Output:
- Overall joint statistics and recommendations, plus per-slide summaries and issues

Notebook step: "Gesture Analysis (Overall)".

Note:
- `agents/tools/gesture_tool.py` imports `GestureConfig` and `analyze_gestures_from_video` from `video_analysis/gesture_analysis.py`, but that module currently defines only `Gestures`. Keep this in mind if you see import errors during execution.

## 12) Prepare NeMo Payload

File:
- `outputs/analysis_payload_for_nemo.json`

This file merges:
- `payload` (speech + slides + OCR)
- `results` (face cache + clothing + emotion + gaze + gesture)

Notebook step: "Recommendations (NeMo ReAct)" (payload write happens here).

## 13) NeMo ReAct Recommendations

NAT package:
- `speaker_feedback_nemo/`
  - `tools.py`: `load_payload`, `slim_payload`, `get_slide_context`
  - `register.py`: NAT tool registration
  - `configs/recommendations.yml`: ReAct workflow config

Tools available to NeMo:
- `load_payload`: loads and trims the JSON payload
- `get_slide_context`: fetches a single slide with OCR/visual context and emotion/gaze/gesture summaries
  - `max_slide_text_chars` default: 800 (for visual/spoken text trimming)

Agent config:
- LLM: `speaker_llm` (NVIDIA NIM)
- LLM params: `temperature=0.2`, `top_p=0.95`, `max_tokens=2048`
- Workflow: `react_agent`
- Output requirement: JSON only
- Max iterations: 6, max tool calls: 20

Expected NeMo output:
- `outputs/nemo_recommendations.json` with:
  - `overall`: list of recommendation strings
  - `per_slide`: dict of slide_id -> `strengths` + `improvements`

## 14) Final Report Export

Output files:
- `outputs/presentation_report.md`
- `outputs/presentation_report.json`

Markdown report includes:
- Overall recommendations
- Slide-by-slide sections with OCR text, tables/figures, strengths, improvements

Notebook step: "Export Markdown Report".

## 15) PDF Export (Optional)

The markdown report can be converted to PDF using your preferred tooling (pandoc or a Markdown-to-PDF pipeline).

## Summary of Thresholds (Quick Reference)

Speech analysis:
- Speech rate window: 60s
- Slow < 60 wpm, fast > 140 wpm
- Background noise threshold: 0.45 (30s window)
- Intelligibility segment length: 30s (notebook override)

Slide detection:
- Detectron score_thresh: 0.95
- SSIM threshold: 0.70 (notebook override)
- Debounce: 1.0s
- Min segment: 10.0s (notebook override)
- OCR similarity_threshold: 0.78
- OCR min_word_count_for_slide: 15

Face cache (MTCNN):
- prob_thresh: 0.5
- mtcnn_thresholds: (0.3, 0.4, 0.5)
- min_face_size: 20

Clothing:
- frames_per_slide_max: 4
- min_face_conf: 0.55

Emotion:
- frames_per_slide_max: 6
- min_face_conf: 0.55
- expand_scale: 1.25
- min_face_size: 48

Gaze:
- frames_per_slide_max: 6
- min_face_conf: 0.55
- expand_scale: 1.3
- min_face_size: 48
- valid_gaze_ratio threshold: 0.3
- left/right dominance: > 40%

Gesture:
- YOLO conf: 0.3
- Joint keypoint conf: 0.30
- Movement thresholds: < 6 low, > 18 high
