"""
model.py — CLIP-based aesthetic scorer.

AestheticScorer encodes outfit images and aesthetic text prompts into the
shared CLIP embedding space, then computes cosine similarity to produce
a 0-10 rating.

Scoring intuition
-----------------
CLIP cosine similarities between a fashion photo and a matching style
description typically fall in [0.20, 0.38].  We linearly map this range
onto [0, 10] and clamp the result, giving:

    score 0   → similarity ≤ 0.18  (completely unrelated)
    score 5   → similarity ≈ 0.28  (moderate match)
    score 10  → similarity ≥ 0.38  (strong match)
"""

import os
import numpy as np
from PIL import Image

import torch
from transformers import CLIPProcessor, CLIPModel

# Calibration: map cosine-sim range to [0, 10]
_SIM_LOW  = 0.18   # floor  → score 0
_SIM_HIGH = 0.38   # ceiling → score 10


class AestheticScorer:
    """
    Rates how closely an outfit photo matches a named aesthetic.

    Parameters
    ----------
    model_name : str
        HuggingFace CLIP model id (default openai/clip-vit-base-patch32).
    device : str | None
        "cuda" / "mps" / "cpu"; auto-detected when None.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str | None = None,
    ):
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device
        print(f"[AestheticScorer] Loading CLIP on {device} …")
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
        print("[AestheticScorer] Ready.")

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    def _text_features(self, inputs) -> torch.Tensor:
        """
        Extract text feature tensor from CLIP, compatible with transformers v4 and v5.
        v4: get_text_features() returns a Tensor directly.
        v5: get_text_features() returns BaseModelOutputWithPooling; we go through
            text_model + text_projection manually.
        """
        out = self.model.get_text_features(**inputs)
        if isinstance(out, torch.Tensor):
            return out
        # v5 path: use text_model pooler_output + text_projection
        text_out = self.model.text_model(
            input_ids=inputs.get("input_ids"),
            attention_mask=inputs.get("attention_mask"),
        )
        feat = self.model.text_projection(text_out.pooler_output)
        return feat

    def _image_features(self, inputs) -> torch.Tensor:
        """Extract image feature tensor, compatible with transformers v4 and v5."""
        out = self.model.get_image_features(**inputs)
        if isinstance(out, torch.Tensor):
            return out
        # v5 path
        vision_out = self.model.vision_model(pixel_values=inputs["pixel_values"])
        feat = self.model.visual_projection(vision_out.pooler_output)
        return feat

    def encode_image(self, image: Image.Image) -> np.ndarray:
        """Return L2-normalised CLIP image embedding (shape [512])."""
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            feat = self._image_features(inputs)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.squeeze().cpu().numpy()

    def encode_text(self, text: str) -> np.ndarray:
        """Return L2-normalised CLIP text embedding (shape [512])."""
        inputs = self.processor(
            text=[text], return_tensors="pt", padding=True
        ).to(self.device)
        with torch.no_grad():
            feat = self._text_features(inputs)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.squeeze().cpu().numpy()

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """Batch-encode multiple texts, return shape [N, 512]."""
        inputs = self.processor(
            text=texts, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        with torch.no_grad():
            feat = self._text_features(inputs)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.cpu().numpy()

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        """Batch-encode multiple PIL images, return shape [N, 512]."""
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            feat = self._image_features(inputs)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.cpu().numpy()

    # ------------------------------------------------------------------
    # Prototype building
    # ------------------------------------------------------------------

    def build_text_prototype(self, prompts: list[str]) -> np.ndarray:
        """
        Average multiple text embeddings into one unit-norm prototype.
        More prompts = more robust description of the aesthetic.
        """
        embs = self.encode_texts(prompts)          # [N, 512]
        mean = embs.mean(axis=0)
        return mean / np.linalg.norm(mean)

    def build_image_prototype(self, images: list[Image.Image]) -> np.ndarray:
        """Average image embeddings of reference outfit photos into one prototype."""
        if not images:
            raise ValueError("Need at least one reference image.")
        embs = self.encode_images(images)
        mean = embs.mean(axis=0)
        return mean / np.linalg.norm(mean)

    def blend_prototypes(
        self,
        text_proto: np.ndarray,
        image_proto: np.ndarray | None,
        text_weight: float = 0.6,
    ) -> np.ndarray:
        """
        Blend a text prototype with an optional image prototype.
        image_weight = 1 - text_weight.
        """
        if image_proto is None:
            return text_proto
        blended = text_weight * text_proto + (1 - text_weight) * image_proto
        return blended / np.linalg.norm(blended)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    @staticmethod
    def similarity_to_score(
        similarity: float,
        low: float = _SIM_LOW,
        high: float = _SIM_HIGH,
    ) -> float:
        """Map a cosine similarity in [low, high] to a 0-10 score."""
        raw = (similarity - low) / (high - low) * 10.0
        return float(max(0.0, min(10.0, raw)))

    def score_image(
        self,
        image: Image.Image,
        aesthetic_name: str,
        prototype: np.ndarray,
    ) -> dict:
        """
        Rate an outfit image against one aesthetic prototype.

        Returns
        -------
        dict with keys:
            aesthetic   – name of the target aesthetic
            similarity  – raw cosine similarity (float in ~[0, 1])
            score       – 0-10 rating (float)
            label       – qualitative label string
        """
        img_emb = self.encode_image(image)
        similarity = float(np.dot(img_emb, prototype))
        score = self.similarity_to_score(similarity)
        label = _score_label(score)

        return {
            "aesthetic": aesthetic_name,
            "similarity": round(similarity, 4),
            "score": round(score, 2),
            "label": label,
        }

    def rank_aesthetics(
        self,
        image: Image.Image,
        prototypes: dict[str, np.ndarray],
    ) -> list[dict]:
        """
        Score an image against every aesthetic prototype and return results
        sorted by score descending.
        """
        img_emb = self.encode_image(image)
        results = []
        for name, proto in prototypes.items():
            sim = float(np.dot(img_emb, proto))
            results.append(
                {
                    "aesthetic": name,
                    "similarity": round(sim, 4),
                    "score": round(self.similarity_to_score(sim), 2),
                    "label": _score_label(self.similarity_to_score(sim)),
                }
            )
        return sorted(results, key=lambda r: r["score"], reverse=True)


# ------------------------------------------------------------------
# Prototype persistence
# ------------------------------------------------------------------

def save_prototypes(prototypes: dict[str, np.ndarray], path: str) -> None:
    """Save a {name: embedding} dict to a .npz file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **prototypes)
    print(f"Prototypes saved → {path}")


def load_prototypes(path: str) -> dict[str, np.ndarray]:
    """Load a .npz prototype file into a {name: embedding} dict."""
    data = np.load(path)
    return {k: data[k] for k in data.files}


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _score_label(score: float) -> str:
    if score >= 8.5:
        return "Perfect match"
    elif score >= 7.0:
        return "Strong match"
    elif score >= 5.5:
        return "Good match"
    elif score >= 4.0:
        return "Moderate match"
    elif score >= 2.5:
        return "Weak match"
    else:
        return "Little to no match"
