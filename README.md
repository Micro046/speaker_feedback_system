# PRISM – Slide-Aligned Multimodal Feedback for Presentation Skills

**PRISM** (Presentation Insights System for Multimodal and Multilingual Feedback) is an end-to-end system that analyzes recorded presentations and generates **slide-aligned, evidence-backed coaching reports**.

It combines:

- speech analysis  
- slide content understanding (detection + OCR)  
- visual delivery cues (gaze, emotion, gesture, attire)  

into a single structured pipeline that produces an actionable PDF report for presenters.

This repository contains the full research prototype used in our CHI 2026 system paper and demo.

---

## Key Features

- **Slide-aligned analysis** – all modalities are synchronized to detected slide segments  
- **Multimodal feedback** – speech, slides, gaze, emotion, gestures, clothing  
- **Evidence-backed metrics** – WPM, fillers, similarity, coverage, timestamps  
- **Two structured reports**
  - Slide + Speech Feedback (storytelling & content alignment)
  - Visual Coaching (delivery behavior)
- **Optional LLM recommendations** via configurable NeMo Agent / OpenRouter
- **Gradio web app** for video → PDF generation
- **Notebook pipeline** for research and debugging

---

## What the User Gets

- A single **PDF coaching report** with:
  - slide-by-slide feedback  
  - extracted slide text  
  - aligned speech excerpts  
  - delivery cues (gaze, gestures, emotions, attire)  
  - quantitative metrics  
  - natural-language recommendations (optional)

- Intermediate JSON payloads for reproducibility and analysis.

---

## System Overview (Pipeline)

1. **Slide segmentation** via frame similarity
2. **Slide region detection** using Detectron2
3. **OCR** using Qwen2.5-VL (or alternatives)
4. **Speech transcription** with Whisper (word timestamps)
5. **Speech metrics** (rate, fillers, noise, intelligibility)
6. **Dominant speaker detection** (MTCNN + embeddings)
7. **Visual cues**
   - Gaze (MediaPipe head-pose)
   - Emotion (EmotiEffNet / DeepFace)
   - Gesture (YOLOv8 pose events)
   - Clothing (CLIP embeddings)
8. **Temporal alignment** to slide segments
9. **Aggregation & reporting**
10. **Optional LLM-based recommendation synthesis**

---

## Repository Structure

```

speaker_feedback_system/
│
├── src/speaker_feedback/        # Core pipeline modules
│   ├── agents/                  # LLM tools & NeMo agents
│   ├── speech/                  # ASR + speech metrics
│   ├── slide/                   # slide detection + OCR
│   ├── video/                   # frame sampling & face cache
│   ├── visual/                  # gaze, emotion, pose, clothing
│   └── tools/
│
├── notebooks/
│   ├── speaker_feedback.ipynb   # Full step-by-step pipeline
│   └── Recommendation.ipynb
│
├── scripts/
│   ├── setup_env.sh
│   └── setup_paperspace.py
│
├── data/
│   ├── inputs/
│   ├── cache/
│   └── outputs/
│
├── gradio_app.py
├── requirements.txt
└── README.md

```

---

## Outputs

After processing a video:

```

data/outputs/
├── content_report_payload.json        # Slide + speech analysis
├── visual_report_payload.json         # Visual modalities
└── gradio/<timestamp>/
└── presentation_feedback_report.pdf

````

Payloads include timestamps, slide IDs, metrics, OCR text, and detected events.

---

## Installation

### 1. Python environment

Python 3.9–3.11 recommended.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
````

GPU is strongly recommended for Detectron2, OCR, and pose models.

---

### 2. Environment variables

Create `.env` in the project root:

```env
OPEN_ROUTER=your_api_key_here
```

The pipeline loads `.env` with `override=True`.

Optional overrides:

```env
DETECTRON_MODEL_PATH=/datasets/model_best/model_best.pth
DETECTRON_CONFIG_PATH=/notebooks/data/cache/my_custom_config.yaml
OCR_MODEL_ID=Qwen/Qwen2.5-VL-3B-Instruct
WHISPER_MODEL_SIZE=small
WHISPER_LANGUAGE=en
SF_OFFLINE_MODE=1
```

---

## Running the Notebook Pipeline

Open:

```
notebooks/speaker_feedback.ipynb
```

This notebook:

* runs every modality step-by-step
* shows intermediate outputs
* builds both JSON payloads
* generates the combined PDF report

Recommended for research, debugging, and understanding the system.

---

## Running the Gradio App (Video → PDF)

### 1. Warm up model caches (optional but recommended)

```bash
uv run python scripts/setup_paperspace.py
```

### 2. Start the app

```bash
uv run python gradio_app.py
```

Then open the printed local URL in your browser.

The app:

* shows live progress across modalities
* saves all artifacts
* generates a single coaching PDF

---

## Models Used

| Task            | Models                                   |
| --------------- | ---------------------------------------- |
| ASR             | Whisper                                  |
| Slide detection | Detectron2 (custom trained)              |
| OCR             | Qwen2.5-VL (default), EasyOCR, PaddleOCR |
| Face detection  | MTCNN                                    |
| Gaze            | MediaPipe head-pose                      |
| Emotion         | EmotiEffNet, DeepFace                    |
| Pose / gesture  | YOLOv8-pose                              |
| Clothing        | CLIP                                     |
| LLM (optional)  | OpenRouter via NeMo Agent Toolkit        |

---

## Dataset for Slide Detector

Custom dataset:

579 annotated frames from online and stage presentations.

Details:
[https://github.com/Micro046/speaker_feedback_system/blob/master/Slides_Dataset.md](https://github.com/Micro046/speaker_feedback_system/blob/master/Slides_Dataset.md)

---

## Demo

YouTube demo:

[https://www.youtube.com/watch?v=7pFJXnWfZ_c](https://www.youtube.com/watch?v=7pFJXnWfZ_c)

---

## License

Apache License 2.0 – see `LICENSE`.

---

## Acknowledgements

Developed by:

* Hassan Iftikhar
* Ali Arshad

Under the supervision of:

* Andrey Savchenko (Sber AI Lab)
