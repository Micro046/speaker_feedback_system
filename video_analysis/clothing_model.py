# video_analysis/clothing_model.py
from __future__ import annotations

import os
import numpy as np
import torch
from typing import List, Dict, Any, Optional
from PIL import Image

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

        # Prompts tuned for presentation coaching (you can expand later)
        self.labels = [
            "formal business attire",
            "business casual attire",
            "casual attire",
            "revealing outfit",
            "inappropriate outfit for a professional presentation",
            "professional outfit suitable for presenting",
        ]

    def assess_appearance(self, frames_rgb: List[np.ndarray], *, return_full: bool = True) -> Dict[str, Any]:
        if not frames_rgb:
            return {"is_appropriate": None, "captions": [], "recommendation": "No frames provided.", "confidence": 0.0}

        images = [Image.fromarray(f) for f in frames_rgb]

        inputs = self.processor(
            text=self.labels,
            images=images,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        with torch.inference_mode():
            outputs = self.model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=-1).detach().cpu().numpy()

        avg = probs.mean(axis=0)
        best = int(np.argmax(avg))
        best_label = self.labels[best]
        conf = float(avg[best])

        # simple decision rule
        inappropriate = {"revealing outfit", "inappropriate outfit for a professional presentation"}
        is_appropriate = best_label not in inappropriate

        recommendation = (
            "Clothing appears appropriate for a professional presentation."
            if is_appropriate else
            "Clothing may be distracting for a professional setting; consider more formal, less revealing attire."
        )

        return {
            "is_appropriate": is_appropriate,
            "captions": [f"{self.labels[i]}: {avg[i]:.2f}" for i in range(len(self.labels))],
            "recommendation": recommendation,
            "confidence": conf,
        }
