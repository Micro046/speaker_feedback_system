# Speaker Feedback System

Prototype pipeline for presentation feedback with two reports:
1) Visual Coaching (delivery style)
2) Slide + Speech Feedback (storytelling and alignment)

## What This Project Does
- Speech analysis: transcription, fillers, speech rate, intelligibility, noise
- Slide analysis: transitions, OCR, description, layout, saved slide crops
- Face cache: dominant-speaker frames for visual modalities
- Clothing: attire profile and appropriateness
- Gaze: head-pose gaze categories (audience/script/slides)
- Emotion: per-slide and overall emotion distribution
- Gesture: semantic body-language events with timestamps

## Project Structure
- `src/speaker_feedback/`: core pipeline (speech, slide, video, tools, NeMo)
- `notebooks/`: step-by-step prototype notebook
- `data/inputs/`: input videos and assets
- `data/cache/`: local model caches
- `data/outputs/`: generated payloads and reports
- `scripts/`: environment setup helpers

## Outputs
- `data/outputs/content_report_payload.json`
  - Slide + speech only, includes evidence intervals and alignment metrics
- `data/outputs/visual_report_payload.json`
  - Visual modalities only, includes multimodal events and timestamps

## Environment
Create a `.env` file in the project root and set:
```
OPENROUTER_API_KEY=your_key_here
```
The notebook loads `.env` with `override=True`.

Optional setup helper:
- `scripts/setup_env.sh` (for CUDA/Jupyter environments)

## Notebook
Use:
- `notebooks/speaker_feedback.ipynb`

This notebook runs the full pipeline step-by-step and shows keys and previews after each modality.
It also builds both payloads and includes optional NeMo recommendation calls.

## NeMo Recommendations
Two prompts are provided:
- Visual Coaching report (multi-modal cues + timestamps)
- Slide + Speech report (content alignment + speech evidence)

In the notebook, set:
- `nemo_config_path` to your NeMo `recommendations.yml`
- `run_nemo = True` to execute

## Notes
- Per-slide report uses only speech + slide content.
- Visual report uses gaze/emotion/gesture/clothing with reliability gates.
- Slide crops are stored under `data/inputs/video/slides/<video_name>/`.
