from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", override=True)
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DEFAULT_DETECTRON_MODEL_PATH = Path("/datasets/model_best/model_best.pth")
DEFAULT_DETECTRON_CONFIG_PATH = Path("/notebooks/data/cache/my_custom_config.yaml")
DEFAULT_OCR_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"


def _log(message: str) -> None:
    print(message, flush=True)


def _resolve_path(env_key: str, default_path: Path) -> Path:
    value = os.getenv(env_key)
    path = Path(value) if value else default_path
    return path


def main() -> None:
    _log("== Paperspace setup: starting ==")

    try:
        import torch
        _log(f"CUDA available: {torch.cuda.is_available()}")
    except Exception as exc:
        _log(f"torch import failed: {repr(exc)}")

    detectron_model_path = _resolve_path("DETECTRON_MODEL_PATH", DEFAULT_DETECTRON_MODEL_PATH)
    detectron_config_path = _resolve_path("DETECTRON_CONFIG_PATH", DEFAULT_DETECTRON_CONFIG_PATH)
    if not detectron_model_path.exists():
        _log(f"[WARN] Detectron model not found: {detectron_model_path}")
    else:
        _log(f"[OK] Detectron model: {detectron_model_path}")
    if not detectron_config_path.exists():
        _log(f"[WARN] Detectron config not found: {detectron_config_path}")
    else:
        _log(f"[OK] Detectron config: {detectron_config_path}")

    from speaker_feedback.config.thresholds import DEFAULT_THRESHOLDS_PATH, load_thresholds

    thresholds = load_thresholds(DEFAULT_THRESHOLDS_PATH)
    slides_cfg = thresholds["slides"]
    clothing_cfg = thresholds["clothing"]
    gesture_cfg = thresholds["gesture"]

    ocr_model_id = os.getenv("OCR_MODEL_ID", DEFAULT_OCR_MODEL_ID)
    cache_dir = PROJECT_ROOT / "data" / "cache" / "model_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    _log("Downloading OCR model (Hugging Face)...")
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=ocr_model_id,
            cache_dir=str(cache_dir),
            local_files_only=False,
        )
        _log(f"[OK] OCR model cached: {ocr_model_id}")
    except Exception as exc:
        _log(f"[WARN] OCR model download failed: {repr(exc)}")

    _log("Downloading Whisper model + NLTK punkt...")
    try:
        import nltk
        nltk.download("punkt", quiet=True)
        import whisper_timestamped

        whisper_model_size = os.getenv("WHISPER_MODEL_SIZE", "small")
        whisper_timestamped.load_model(whisper_model_size)
        _log(f"[OK] Whisper model cached: {whisper_model_size}")
    except Exception as exc:
        _log(f"[WARN] Whisper/NLTK download failed: {repr(exc)}")

    _log("Downloading CLIP model for clothing analysis...")
    try:
        from transformers import CLIPModel, CLIPProcessor

        clip_model_name = clothing_cfg["clip_model_name"]
        CLIPProcessor.from_pretrained(clip_model_name, local_files_only=False)
        CLIPModel.from_pretrained(clip_model_name, local_files_only=False)
        _log(f"[OK] CLIP cached: {clip_model_name}")
    except Exception as exc:
        _log(f"[WARN] CLIP download failed: {repr(exc)}")

    _log("Downloading YOLO pose model...")
    try:
        from ultralytics import YOLO

        model_path = gesture_cfg["model_path"]
        YOLO(model_path)
        _log(f"[OK] YOLO pose model ready: {model_path}")
    except Exception as exc:
        _log(f"[WARN] YOLO download failed: {repr(exc)}")

    _log("Downloading emotion model...")
    try:
        import torch
        from emotiefflib.facial_analysis import EmotiEffLibRecognizer, get_model_list

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = get_model_list()[0]
        EmotiEffLibRecognizer(engine="onnx", model_name=model_name, device=device)
        _log(f"[OK] Emotion model ready: {model_name}")
    except Exception as exc:
        _log(f"[WARN] Emotion model setup failed: {repr(exc)}")

    _log("== Paperspace setup: complete ==")
    _log("If any WARN items appeared, fix paths or run again with internet access.")


if __name__ == "__main__":
    main()
