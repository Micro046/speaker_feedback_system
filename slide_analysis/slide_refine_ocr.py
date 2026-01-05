# slide_analysis/slide_refine_ocr.py
from __future__ import annotations

import gc
import logging
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

logger = logging.getLogger(__name__)

# =========================================================
# Text normalization & tokenization
# =========================================================
_WORD_RE = re.compile(r"[a-z0-9]+")

def normalize_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def tokenize(s: str) -> List[str]:
    return _WORD_RE.findall(normalize_text(s))


# =========================================================
# Similarity metrics
# =========================================================
def jaccard_tokens(a: List[str], b: List[str]) -> float:
    A, B = set(a), set(b)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)

def edit_sim_prefix(a: str, b: str, n: int = 200) -> float:
    a = normalize_text(a)[:n]
    b = normalize_text(b)[:n]
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return float(SequenceMatcher(None, a, b).ratio())

def combined_similarity(
    text_a: str,
    text_b: str,
    *,
    min_token_count_for_similarity: int = 5,
    edit_prefix_chars: int = 200,
    w_jaccard: float = 0.6,
    w_edit: float = 0.4,
) -> float:
    ta = tokenize(text_a)
    tb = tokenize(text_b)

    if len(ta) < min_token_count_for_similarity and len(tb) < min_token_count_for_similarity:
        return 1.0

    j = jaccard_tokens(ta, tb)
    e = edit_sim_prefix(text_a, text_b, n=edit_prefix_chars)
    return w_jaccard * j + w_edit * e


# =========================================================
# OCR cleanup & quality scoring (CRITICAL)
# =========================================================
def dedupe_lines(text: str, max_repeat: int = 2) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    seen = {}
    out = []
    for ln in lines:
        key = ln.lower()
        seen[key] = seen.get(key, 0) + 1
        if seen[key] <= max_repeat:
            out.append(ln)
    return "\n".join(out)

def clamp_repeated_ngrams(text: str, n: int = 3, max_repeat: int = 3) -> str:
    toks = tokenize(text)
    if len(toks) < n * 2:
        return text

    seen = {}
    out = []
    i = 0
    while i < len(toks):
        gram = tuple(toks[i:i+n])
        if len(gram) < n:
            out.extend(toks[i:])
            break

        seen[gram] = seen.get(gram, 0) + 1
        if seen[gram] <= max_repeat:
            out.append(toks[i])
        i += 1

    return " ".join(out)

def ocr_cleanup(text: str) -> str:
    if not text:
        return ""
    text = normalize_text(text)
    text = text.replace("\u2022", "\n").replace(" - ", "\n")
    text = dedupe_lines(text, max_repeat=2)
    text = clamp_repeated_ngrams(text, n=3, max_repeat=3)
    return normalize_text(text)

def ocr_quality_score(text: str) -> float:
    toks = tokenize(text)
    if not toks:
        return 0.0
    uniq = len(set(toks))
    total = len(toks)
    repetition_penalty = total - uniq
    return float(uniq) - 2.5 * repetition_penalty


# =========================================================
# Config
# =========================================================
@dataclass
class OCRRefineConfig:
    ocr_model_id: str = "JackChew/Qwen2-VL-2B-OCR"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    similarity_threshold: float = 0.78
    min_token_count_for_similarity: int = 5
    edit_prefix_chars: int = 200
    w_jaccard: float = 0.6
    w_edit: float = 0.4

    min_word_count_for_slide: int = 15

    ocr_prompt: str = (
        "Extract all clearly readable text from this slide. "
        "Return plain text only. "
        "Do not repeat content. "
        "Do not add explanations."
    )


# =========================================================
# Refiner
# =========================================================
class SlideOCRRefiner:
    def __init__(self, predictor, config: Optional[OCRRefineConfig] = None):
        self.predictor = predictor
        self.cfg = config or OCRRefineConfig()

        self.processor = None
        self.model = None
        self._ocr_cache: Dict[int, str] = {}

    # ---------------- OCR lifecycle ----------------
    def _load_ocr_model(self):
        if self.model is not None:
            return

        logger.info("Loading OCR model %s on %s", self.cfg.ocr_model_id, self.cfg.device)

        self.processor = AutoProcessor.from_pretrained(
            self.cfg.ocr_model_id,
            # size={"shortest_edge": 56 * 56, "longest_edge": 28 * 28 * 1280},
        )

        dtype = torch.bfloat16 if self.cfg.device.startswith("cuda") else torch.float32
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.cfg.ocr_model_id,
            torch_dtype=dtype,
        ).to(self.cfg.device)

        self.model.eval()

    def unload(self):
        if self.model is not None:
            logger.info("Unloading OCR model")
            try:
                del self.model
                del self.processor
            except Exception:
                pass
            self.model = None
            self.processor = None

        self._ocr_cache.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---------------- OCR helpers ----------------
    def _crop_slide_region(self, frame_bgr: np.ndarray) -> np.ndarray:
        from slide_analysis.slide_transition_ssim import detect_bbox_on_frame, crop_slide
        bbox = detect_bbox_on_frame(frame_bgr, self.predictor, downscale_max_side=960)
        if bbox is None:
            return frame_bgr
        return crop_slide(frame_bgr, bbox)

    def _ocr_frame(self, frame_bgr: np.ndarray) -> str:
        self._load_ocr_model()

        crop = self._crop_slide_region(frame_bgr)
        pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

        conversation = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": self.cfg.ocr_prompt}
            ]
        }]

        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(
            text=[prompt],
            images=[pil],
            padding=True,
            return_tensors="pt",
        ).to(self.cfg.device)

        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, max_new_tokens=512)

        gen = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
        raw = self.processor.batch_decode(gen, skip_special_tokens=True)[0]
        return ocr_cleanup(raw)

    def _ocr_at_time(self, cap: cv2.VideoCapture, t: float, fps: float) -> str:
        key = int(round(t * 1000))
        if key in self._ocr_cache:
            return self._ocr_cache[key]

        frame_idx = int(round(t * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            self._ocr_cache[key] = ""
            return ""

        txt = self._ocr_frame(frame)
        self._ocr_cache[key] = txt
        return txt


    # ---------------- main refinement ----------------
    def refine_segments(self, video_path: str, ssim_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not ssim_segments:
            return []

        cap = cv2.VideoCapture(str(Path(video_path).resolve()))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0


        refined: List[Dict[str, Any]] = []
        current: Optional[Dict[str, Any]] = None

        for seg in ssim_segments:
            start = float(seg["start_time"])
            end = float(seg["end_time"])
            dur = max(0.001, end - start)

            # Multi-sample OCR (robust)
            sample_times = [
                start + 0.10 * dur,
                (start + end) / 2.0,
                end - 0.10 * dur,
            ]

            best_text = ""
            best_score = -1e9
            best_time = sample_times[1]

            for t in sample_times:
                txt = self._ocr_at_time(cap, t,fps)
                sc = ocr_quality_score(txt)
                if sc > best_score:
                    best_text = txt
                    best_score = sc
                    best_time = t

            word_count = len(tokenize(best_text))

            seg_data = dict(seg)
            seg_data.update({
                "ocr_text": best_text,
                "ocr_word_count": word_count,
                "ocr_best_time": best_time,
                "ocr_quality": best_score,
            })

            if current is None:
                current = seg_data
                continue

            sim = combined_similarity(
                current.get("ocr_text", ""),
                best_text,
                min_token_count_for_similarity=self.cfg.min_token_count_for_similarity,
                edit_prefix_chars=self.cfg.edit_prefix_chars,
                w_jaccard=self.cfg.w_jaccard,
                w_edit=self.cfg.w_edit,
            )

            if sim >= self.cfg.similarity_threshold:
                current["end_time"] = seg_data["end_time"]
                current["duration"] = float(current["end_time"]) - float(current["start_time"])

                if len(best_text) > len(current.get("ocr_text", "")):
                    current.update({
                        "ocr_text": best_text,
                        "ocr_word_count": word_count,
                        "ocr_best_time": best_time,
                        "ocr_quality": best_score,
                    })

            elif word_count < self.cfg.min_word_count_for_slide:
                current["end_time"] = seg_data["end_time"]
                current["duration"] = float(current["end_time"]) - float(current["start_time"])
            else:
                refined.append(current)
                current = seg_data

        if current:
            refined.append(current)

        cap.release()

        for i, s in enumerate(refined, start=1):
            s["slide_id"] = i

        return refined
