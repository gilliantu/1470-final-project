"""
model.py — CLIP-based aesthetic scorer.

AestheticScorer encodes outfit images and aesthetic text prompts into the
shared CLIP embedding space, then uses within-image z-score normalization
to produce 0-10 ratings.

Scoring approach
----------------
CLIP text embeddings for different aesthetics are highly similar to each
other (cosine sim ~0.75) because they cluster in a narrow cone of the
embedding space.  Raw similarities therefore lack dynamic range — e.g. all
aesthetics for a given outfit might fall in [0.44, 0.65].

To produce meaningful 0-10 scores we use **within-image z-score scoring**:

1. Compute the raw cosine similarity between the outfit and every aesthetic.
2. For that image, compute the mean and std across all aesthetics.
3. Each aesthetic's score = 5 + (sim - mean) / std × Z_SCALE
   clamped to [0, 10].

This means:
  • The average-matching aesthetic → ~5 / 10
  • An aesthetic 1 std above average → ~5 + Z_SCALE
  • Z_SCALE = 2.0 → ±2.5 std maps to the full 0-10 range

When scoring against a *single* aesthetic, we still compute all prototypes
internally so the image-specific mean/std are accurate.
"""

import os
import numpy as np
from PIL import Image

import torch
from transformers import CLIPProcessor, CLIPModel

# Controls how much score spread within-image z-scores produce.
# z=+2.5 → score 10,  z=-2.5 → score 0  (each std ≈ 2 points)
_Z_SCALE = 2.0


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
    def _zscore_scores(
        similarities: dict[str, float],
        z_scale: float = _Z_SCALE,
    ) -> dict[str, float]:
        """
        Convert a {aesthetic: raw_cosine_similarity} dict to {aesthetic: score}
        using within-image z-score normalization.

        score = 5 + (sim - mean_sim) / std_sim × z_scale,  clamped [0, 10].

        If std_sim is near zero (all aesthetics match equally), everything
        returns 5.0.
        """
        vals = list(similarities.values())
        mean_s = float(np.mean(vals))
        std_s = float(np.std(vals))
        if std_s < 1e-6:
            return {k: 5.0 for k in similarities}
        scores = {}
        for k, s in similarities.items():
            z = (s - mean_s) / std_s
            scores[k] = float(max(0.0, min(10.0, 5.0 + z * z_scale)))
        return scores

    def rank_aesthetics(
        self,
        image: Image.Image,
        prototypes: dict[str, np.ndarray],
    ) -> list[dict]:
        """
        Score an image against every aesthetic prototype.

        Scores are within-image z-score normalised so the spread across
        aesthetics is meaningful regardless of absolute similarity levels.

        Returns a list of dicts sorted by score descending, each with:
            aesthetic   – aesthetic key
            similarity  – raw cosine similarity
            score       – 0-10 z-score-normalised rating
            label       – qualitative label string
        """
        img_emb = self.encode_image(image)
        raw_sims = {name: float(np.dot(img_emb, proto))
                    for name, proto in prototypes.items()}
        scores = self._zscore_scores(raw_sims)

        results = [
            {
                "aesthetic": name,
                "similarity": round(raw_sims[name], 4),
                "score": round(scores[name], 2),
                "label": _score_label(scores[name]),
            }
            for name in raw_sims
        ]
        return sorted(results, key=lambda r: r["score"], reverse=True)

    def score_image(
        self,
        image: Image.Image,
        aesthetic_name: str,
        prototypes: dict[str, np.ndarray],
    ) -> dict:
        """
        Rate an outfit image against a specific aesthetic.

        Internally ranks all aesthetics so the z-score is calibrated
        against the full prototype set.

        Returns
        -------
        dict with keys:
            aesthetic   – name of the target aesthetic
            similarity  – raw cosine similarity
            score       – 0-10 z-score-normalised rating
            label       – qualitative label string
        """
        ranked = self.rank_aesthetics(image, prototypes)
        for r in ranked:
            if r["aesthetic"] == aesthetic_name:
                return r
        raise ValueError(f"Aesthetic '{aesthetic_name}' not found in prototypes.")


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
