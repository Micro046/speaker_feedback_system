# slide_analysis/slide_refine_ocr.py
from __future__ import annotations

import gc
import logging
import re
import hashlib
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    ocr_model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    similarity_threshold: float = 0.78
    min_token_count_for_similarity: int = 5
    edit_prefix_chars: int = 200
    w_jaccard: float = 0.6
    w_edit: float = 0.4

    min_word_count_for_slide: int = 15
    ocr_sample_count: int = 3
    top_k_ocr_frames: int = 2

    # PHASE 1: Strict OCR prompt (Logic)
    ocr_prompt_strict: str = (
        "Transcribe text exactly as it appears. Return plain text only. "
        "Do not add descriptions. Do not repeat lines."
    )

    # PHASE 2a: Slide description (content for recommendations)
    desc_prompt: str = (
        "Summarize the slide for feedback. Include:\n"
        "1. Content Summary (1-2 sentences)\n"
        "2. Key Takeaway (short)\n"
        "3. Layout (structure and placement)\n\n"
        "Format:\n"
        "[[DESC_START]]\n(text)\n[[DESC_END]]\n"
        "[[TAKEAWAY_START]]\n(text)\n[[TAKEAWAY_END]]\n"
        "[[LAYOUT_START]]\n(text)\n[[LAYOUT_END]]"
    )

    # Best-frame selection (combine OCR quality + sharpness)
    sharpness_weight: float = 0.15


# =========================================================
# Refiner
# =========================================================
class SlideOCRRefiner:
    def __init__(self, predictor, config: Optional[OCRRefineConfig] = None):
        self.predictor = predictor
        self.cfg = config or OCRRefineConfig()

        self.processor = None
        self.model = None
        self._ocr_cache: Dict[int, Any] = {}
        self._ocr_hash_cache: Dict[str, str] = {}

    # ---------------- OCR lifecycle ----------------
    def _load_ocr_model(self):
        if self.model is not None:
            return

        logger.info("Loading OCR model %s on %s", self.cfg.ocr_model_id, self.cfg.device)

        self.processor = AutoProcessor.from_pretrained(
            self.cfg.ocr_model_id,
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
        self._ocr_hash_cache.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---------------- OCR helpers ----------------
    def _crop_slide_region(self, frame_bgr: np.ndarray) -> np.ndarray:
        from speaker_feedback.slide_analysis.slide_transition_ssim import detect_bbox_on_frame, crop_slide
        bbox = detect_bbox_on_frame(frame_bgr, self.predictor, downscale_max_side=960)
        if bbox is None:
            return frame_bgr
        return crop_slide(frame_bgr, bbox)

    def _generate_content(self, frame_bgr: np.ndarray, prompt: str) -> str:
        """Generic generation helper."""
        self._load_ocr_model()
        crop = self._crop_slide_region(frame_bgr)
        pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        
        conversation = [{
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": prompt}]
        }]
        
        prompt_text = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(text=[prompt_text], images=[pil], padding=True, return_tensors="pt").to(self.cfg.device)
        
        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, max_new_tokens=512)
            
        gen = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
        return self.processor.batch_decode(gen, skip_special_tokens=True)[0]

    def _generate_content_from_crop(self, crop_bgr: np.ndarray, prompt: str) -> str:
        """Generation helper using a pre-cropped image."""
        self._load_ocr_model()
        pil = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))

        conversation = [{
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": prompt}]
        }]

        prompt_text = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(text=[prompt_text], images=[pil], padding=True, return_tensors="pt").to(self.cfg.device)

        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, max_new_tokens=512)

        gen = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
        return self.processor.batch_decode(gen, skip_special_tokens=True)[0]

    def _hash_image(self, crop_bgr: np.ndarray) -> str:
        h = hashlib.sha256()
        h.update(crop_bgr.tobytes())
        h.update(str(crop_bgr.shape).encode("utf-8"))
        h.update(str(crop_bgr.dtype).encode("utf-8"))
        return h.hexdigest()

    def _get_crop_at_time(self, cap: cv2.VideoCapture, t: float, fps: float) -> Optional[np.ndarray]:
        key = int(round(t * 1000))
        if key in self._ocr_cache:
            val = self._ocr_cache[key]
            return val.get("crop")

        frame_idx = int(round(t * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            return None

        crop = self._crop_slide_region(frame)
        self._ocr_cache[key] = {"crop": crop}
        return crop

    def _strict_ocr_from_crop(self, crop_bgr: np.ndarray) -> str:
        """Phase 1: Strict OCR on a cropped slide (cached by image hash)."""
        if crop_bgr is None or crop_bgr.size == 0:
            return ""

        key = self._hash_image(crop_bgr)
        cached = self._ocr_hash_cache.get(key)
        if cached is not None:
            return cached

        raw = self._generate_content_from_crop(crop_bgr, self.cfg.ocr_prompt_strict)
        text = ocr_cleanup(raw)
        self._ocr_hash_cache[key] = text
        return text

    def _parse_tag(self, text: str, tag: str) -> str:
        pattern = rf"\\[\\[\\s*{tag}\\s*_START\\s*\\]\\](.*?)\\[\\[\\s*{tag}\\s*_END\\s*\\]\\]"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else ""

    def _strip_tag_markers(self, text: str) -> str:
        if not text:
            return ""
        cleaned = re.sub(r"\\[\\[\\s*[^\\]]+\\s*\\]\\]", "", text, flags=re.IGNORECASE)
        return normalize_text(cleaned)

    def _sharpness_score(self, frame_bgr: np.ndarray) -> float:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def _strict_ocr(self, cap: cv2.VideoCapture, t: float, fps: float) -> Tuple[str, Any]:
        """Phase 1: Fast, strict OCR for logic."""
        crop = self._get_crop_at_time(cap, t, fps)
        if crop is None:
            return "", None
        text = self._strict_ocr_from_crop(crop)
        return text, crop

    def _rich_analysis(self, image_path: str) -> Dict[str, str]:
        """Phase 2: Rich description + layout for final slides."""
        if not Path(image_path).exists():
            return {"description": "", "key_takeaway": "", "layout": ""}
            
        frame = cv2.imread(image_path)
        if frame is None:
            return {"description": "", "key_takeaway": "", "layout": ""}
            
        self._load_ocr_model()
        # image_path is already cropped, so use directly
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
         
        conversation = [{
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": self.cfg.desc_prompt}]
        }]
        
        prompt_text = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(text=[prompt_text], images=[pil], padding=True, return_tensors="pt").to(self.cfg.device)
        
        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, max_new_tokens=512)
            
        raw_output = self.processor.batch_decode([out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)], skip_special_tokens=True)[0]
        
        desc = self._parse_tag(raw_output, "DESC")
        take = self._parse_tag(raw_output, "TAKEAWAY")
        layout = self._parse_tag(raw_output, "LAYOUT")
        if not desc:
            desc = self._strip_tag_markers(raw_output)
        if not take:
            take = ""
        if not layout:
            layout = self._strip_tag_markers(raw_output)
        return {"description": desc, "key_takeaway": take, "layout": layout}


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

        logger.info("Starting Phase 1: Refining segments with Strict OCR")
        for seg in ssim_segments:
            start = float(seg["start_time"])
            end = float(seg["end_time"])
            dur = max(0.001, end - start)

            # Multi-sample (sharpness first), OCR only on top-K frames
            sample_count = max(3, int(self.cfg.ocr_sample_count))
            t_start = start + 0.10 * dur
            t_end = end - 0.10 * dur
            if t_end <= t_start:
                t_start, t_end = start, end
            sample_times = np.linspace(t_start, t_end, sample_count).tolist()

            best_text = ""
            best_score = -1e9
            best_time = (start + end) / 2.0
            best_crop = None
            best_quality = -1e9
            best_sharpness = 0.0

            candidates: List[Tuple[float, float, np.ndarray]] = []
            for t in sample_times:
                crop = self._get_crop_at_time(cap, t, fps)
                if crop is None or crop.size == 0:
                    continue
                sharp = self._sharpness_score(crop)
                candidates.append((sharp, float(t), crop))

            if candidates:
                candidates.sort(key=lambda x: x[0], reverse=True)
                top_k = max(1, int(self.cfg.top_k_ocr_frames))
                for sharp, t, crop in candidates[:top_k]:
                    txt = self._strict_ocr_from_crop(crop)
                    sc = ocr_quality_score(txt)
                    combined = sc + (self.cfg.sharpness_weight * np.log1p(sharp))
                    if combined > best_score:
                        best_text = txt
                        best_score = combined
                        best_time = t
                        best_crop = crop
                        best_quality = sc
                        best_sharpness = sharp

            word_count = len(tokenize(best_text))

            seg_data = dict(seg)
            seg_data.update({
                "ocr_text": best_text,
                "ocr_word_count": word_count,
                "ocr_best_time": best_time,
                "ocr_quality": best_quality,
                "ocr_sharpness": round(float(best_sharpness), 3),
                "best_crop_img": best_crop,
                # Placeholders for Phase 2
                "description": "",
                "key_takeaway": "",
                "layout": ""
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
                        "ocr_quality": best_quality,
                        "ocr_sharpness": round(float(best_sharpness), 3),
                        "best_crop_img": best_crop,
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

        # Prepare output directory
        vid_path_obj = Path(video_path).resolve()
        video_name = vid_path_obj.stem
        output_dir = vid_path_obj.parent / "slides" / video_name
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Starting Phase 2: Rich Analysis of %d unique slides", len(refined))
        for i, s in enumerate(refined, start=1):
            s["slide_id"] = i
            
            # Save the best image
            best_crop = s.get("best_crop_img")
            if best_crop is not None:
                img_name = f"slide_{i:03d}.jpg"
                img_path = output_dir / img_name
                try:
                    cv2.imwrite(str(img_path), best_crop)
                    s["image_path"] = str(img_path)
                except Exception as e:
                    logger.error("Failed to save slide image %s: %s", img_path, e)
                    s["image_path"] = None
                
                # Cleanup heavy array
                del s["best_crop_img"]

                # Run Rich Analysis if image exists
                if s["image_path"]:
                    try:
                        rich = self._rich_analysis(s["image_path"])
                        s["description"] = rich["description"]
                        s["key_takeaway"] = rich["key_takeaway"]
                        s["layout"] = rich.get("layout", "")
                    except Exception as e:
                        logger.error("Rich analysis failed for slide %d: %s", i, e)
            else:
                 s["image_path"] = None

        return refined
