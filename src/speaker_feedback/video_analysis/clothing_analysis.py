from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image


def _force_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class ClothesCLIP:
    """
    Strict CLIP loader: if CLIP fails to load, it raises an error
    (so you see the real reason instead of silently falling back).
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None,
        local_files_only: bool = False,   # set True if you already downloaded the model
        prefer_safetensors: bool = True,
        verbose: bool = True,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.local_files_only = local_files_only
        self.verbose = verbose

        try:
            from transformers import CLIPProcessor, CLIPModel
        except Exception as e:
            raise RuntimeError(
                "transformers CLIP import failed. Install/upgrade transformers.\n"
                f"Original error: {repr(e)}"
            )

        try:
            from packaging.version import Version
        except Exception:
            Version = None

        try:
            if self.verbose:
                print(
                    f"[ClothesCLIP] Loading {model_name} on {self.device} "
                    f"(local_files_only={local_files_only}, prefer_safetensors={prefer_safetensors})"
                )

            self.processor = CLIPProcessor.from_pretrained(
                model_name,
                local_files_only=local_files_only,
            )

            def _load_model(use_safetensors: bool):
                return CLIPModel.from_pretrained(
                    model_name,
                    local_files_only=local_files_only,
                    use_safetensors=use_safetensors,
                ).to(self.device)

            try:
                self.model = _load_model(use_safetensors=prefer_safetensors)
            except Exception as e:
                torch_ok = True
                if Version is not None:
                    try:
                        torch_ok = Version(torch.__version__.split("+")[0]) >= Version("2.6.0")
                    except Exception:
                        torch_ok = True

                if prefer_safetensors and torch_ok:
                    self.model = _load_model(use_safetensors=False)
                else:
                    raise e

            self.model.eval()

        except Exception as e:
            raise RuntimeError(
                "CLIP failed to load. Most common causes:\n"
                " - No internet / HF blocked (set local_files_only=True after downloading)\n"
                " - Cache dir not writable\n"
                " - transformers/tokenizers/safetensors mismatch\n"
                " - torch < 2.6 requires safetensors weights\n"
                f"\nOriginal error: {repr(e)}"
            )

        # Prompts tuned for presentation coaching
        self.prompt_groups = {
            "style": [
                "formal business suit",
                "smart casual business attire",
                "casual street clothing",
                "sportswear or athletic clothing",
                "distracting or revealing outfit"
            ],
            "top_type": [
                "suit jacket or blazer",
                "button-down dress shirt",
                "polo shirt",
                "t-shirt",
                "hoodie or sweatshirt",
                "tank top"
            ],
            "pattern": [
                "solid color clothing",
                "striped pattern clothing",
                "plaid or checkered pattern",
                "graphic print or logo"
            ],
            "color_tone": [
                "dark colored clothing (black, navy, charcoal)",
                "light colored clothing (white, beige, light gray)",
                "bright colorful clothing (red, yellow, green)"
            ]
        }

    def assess_appearance(self, frames_rgb: List[np.ndarray], *, return_full: bool = True) -> Dict[str, Any]:
        if not frames_rgb:
            return {
                "is_appropriate": None,
                "captions": [],
                "recommendation": "No frames provided.",
                "confidence": 0.0,
            }

        images = [Image.fromarray(f) for f in frames_rgb]

        # Flatten all prompts to run in one batch (efficient)
        all_labels = []
        group_slices = {}
        curr_idx = 0

        # Order matters for consistent slicing
        group_names = ["style", "top_type", "pattern", "color_tone"]

        for name in group_names:
            prompts = self.prompt_groups[name]
            all_labels.extend(prompts)
            group_slices[name] = (curr_idx, curr_idx + len(prompts))
            curr_idx += len(prompts)

        inputs = self.processor(
            text=all_labels,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)

        with torch.inference_mode():
            outputs = self.model(**inputs)
            logits = outputs.logits_per_image

        avg_logits = logits.mean(dim=0).detach().cpu()

        results = {}
        description_parts = []

        for name in group_names:
            start, end = group_slices[name]
            sub_logits = avg_logits[start:end]
            sub_probs = sub_logits.softmax(dim=0).numpy()

            best_idx = int(np.argmax(sub_probs))
            best_label = self.prompt_groups[name][best_idx]
            confidence = float(sub_probs[best_idx])

            results[name] = {
                "label": best_label,
                "confidence": confidence,
                "probs": {l: float(p) for l, p in zip(self.prompt_groups[name], sub_probs)}
            }

            if confidence > 0.4:
                short_label = best_label.replace("clothing", "").replace("attire", "").strip()
                description_parts.append(short_label)

        style_res = results["style"]
        is_appropriate = True
        recommendation = "Professional appearance."

        bad_styles = [
            "casual street",
            "sportswear",
            "distracting"
        ]

        if any(bad in style_res["label"] for bad in bad_styles):
            is_appropriate = False
            recommendation = f"Detected {style_res['label']}. Consider more formal attire for better professional impact."
        elif "smart casual" in style_res["label"] and results["top_type"]["label"] == "hoodie or sweatshirt":
            is_appropriate = False
            recommendation = "Smart casual context detected, but hoodies are generally too casual."

        full_description = ", ".join(description_parts)

        return {
            "is_appropriate": is_appropriate,
            "description": full_description,
            "style": style_res["label"],
            "top": results["top_type"]["label"],
            "recommendation": recommendation,
            "confidence": style_res["confidence"],
            "detailed_results": results
        }


@dataclass
class ClothingConfig:
    frames_per_slide_max: int = 4
    min_face_conf: float = 0.55

    torso_scale_w: float = 2.2
    torso_scale_h: float = 3.2
    torso_shift_y: float = 1.15

    read_every_nth: int = 1
    max_total_frames: int = 300


def _clamp_box(x1, y1, x2, y2, w, h):
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(x1 + 1, min(int(x2), w))
    y2 = max(y1 + 1, min(int(y2), h))
    return x1, y1, x2, y2


def torso_crop_from_face_bbox(frame_bgr: np.ndarray, face_bbox: List[int], cfg: ClothingConfig) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = face_bbox
    fw = max(1, x2 - x1)
    fh = max(1, y2 - y1)

    cx = x1 + fw / 2.0
    cy = y1 + fh / 2.0

    torso_w = fw * cfg.torso_scale_w
    torso_h = fh * cfg.torso_scale_h

    # guard: avoid huge background crops
    torso_w = min(torso_w, w * 0.95)
    torso_h = min(torso_h, h * 0.95)

    torso_cy = cy + fh * cfg.torso_shift_y

    tx1 = cx - torso_w / 2.0
    ty1 = torso_cy - torso_h / 2.0
    tx2 = cx + torso_w / 2.0
    ty2 = torso_cy + torso_h / 2.0

    tx1, ty1, tx2, ty2 = _clamp_box(tx1, ty1, tx2, ty2, w, h)
    crop = frame_bgr[ty1:ty2, tx1:tx2]

    # fallback if crop is too small
    if crop.size < 1000:
        fx1, fy1, fx2, fy2 = _clamp_box(x1, y1, x2, y2, w, h)
        return frame_bgr[fy1:fy2, fx1:fx2]

    return crop


def pick_best_frames_per_slide(
    slide_frame_mapping: Dict[int, Dict[str, Any]],
    face_crops_cache: Dict[int, List[Dict[str, Any]]],
    cfg: ClothingConfig,
) -> List[Dict[str, Any]]:
    """
    Returns a list of {slide_id, frame_idx, face_conf, bbox}.
    Picks top-N frames per slide by face area (stable) with confidence >= min_face_conf.
    """
    picked: List[Dict[str, Any]] = []

    for slide_id, info in slide_frame_mapping.items():
        candidates = []
        for idx in info.get("frame_indices", []):
            faces = face_crops_cache.get(idx)
            if not faces:
                continue

            # prefer largest face (speaker) if area is available
            best = max(faces, key=lambda d: float(d.get("area", 0.0)))
            conf = float(best.get("confidence", 0.0))
            bbox = best.get("bbox")

            if bbox is None:
                continue
            if conf >= cfg.min_face_conf:
                candidates.append((float(best.get("area", 0.0)), conf, idx, bbox))

        # sort by area desc, then confidence desc
        candidates.sort(reverse=True, key=lambda x: (x[0], x[1]))

        for _, conf, idx, bbox in candidates[: cfg.frames_per_slide_max]:
            picked.append({
                "slide_id": int(slide_id),
                "frame_idx": int(idx),
                "face_conf": float(conf),
                "bbox": bbox,
            })

    picked.sort(key=lambda r: (r["slide_id"], -r["face_conf"]))
    if len(picked) > cfg.max_total_frames:
        picked = picked[: cfg.max_total_frames]

    return picked


def analyze_clothing(
    video_path: str,
    slide_frame_mapping: Dict[int, Dict[str, Any]],
    face_crops_cache: Dict[int, List[Dict[str, Any]]],
    clothing_classifier,
    *,
    cfg: Optional[ClothingConfig] = None,
) -> Dict[str, Any]:
    cfg = cfg or ClothingConfig()

    picked = pick_best_frames_per_slide(slide_frame_mapping, face_crops_cache, cfg)
    if not picked:
        return {
            "is_appropriate": None,
            "detected_attributes": [],
            "recommendation": "No suitable frames with faces found for clothing analysis.",
            "coverage": {"slides_with_samples": 0, "frames_used": 0},
            "per_slide": {},
        }

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    per_slide_frames_rgb: Dict[int, List[np.ndarray]] = {}
    per_slide_meta: Dict[int, List[Dict[str, Any]]] = {}

    def read_frame(idx: int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        return ok, frame

    for j, rec in enumerate(picked):
        if cfg.read_every_nth > 1 and (j % cfg.read_every_nth != 0):
            continue

        ok, frame_bgr = read_frame(rec["frame_idx"])
        if not ok or frame_bgr is None:
            continue

        torso_bgr = torso_crop_from_face_bbox(frame_bgr, rec["bbox"], cfg)
        if torso_bgr is None or torso_bgr.size == 0:
            continue

        torso_rgb = cv2.cvtColor(torso_bgr, cv2.COLOR_BGR2RGB)

        sid = rec["slide_id"]
        per_slide_frames_rgb.setdefault(sid, []).append(torso_rgb)
        per_slide_meta.setdefault(sid, []).append(rec)

    cap.release()
    _force_cleanup()

    all_crops = []
    crop_owner = []
    for sid, crops in per_slide_frames_rgb.items():
        for c in crops:
            all_crops.append(c)
            crop_owner.append(sid)

    if not all_crops:
        return {
            "is_appropriate": None,
            "detected_attributes": [],
            "recommendation": "Could not decode any sampled frames for clothing analysis.",
            "coverage": {"slides_with_samples": 0, "frames_used": 0},
            "per_slide": {},
        }

    out = clothing_classifier.assess_appearance(all_crops, return_full=True)

    slides_with_samples = len(per_slide_frames_rgb)
    per_slide = {
        str(sid): {
            "frames_used": len(per_slide_frames_rgb[sid]),
            "best_face_conf": max([m["face_conf"] for m in per_slide_meta.get(sid, [])], default=0.0),
        }
        for sid in per_slide_frames_rgb.keys()
    }

    return {
        "is_appropriate": out.get("is_appropriate"),
        "description": out.get("description", ""),
        "style": out.get("style", ""),
        "top": out.get("top", ""),
        "detected_attributes": out.get("captions", []) or [],
        "recommendation": out.get("recommendation", ""),
        "coverage": {
            "slides_with_samples": slides_with_samples,
            "frames_used": len(all_crops),
        },
        "per_slide": per_slide,
        "debug": {
            "picked_frames": picked[:50],
        }
    }
