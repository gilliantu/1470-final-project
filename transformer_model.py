"""
transformer_model.py — Custom transformer for aesthetic scoring.

Architecture
------------
1. CLIP ViT-B/32 (frozen) → 50 patch tokens × 768-dim
   (the raw vision backbone output *before* the 512-dim projection layer)
2. Linear projection: 768 → d_model
3. Learned positional encoding added
4. N × TransformerEncoderLayer  (self-attention + FFN, built from scratch)
5. CLS token representation → MLP head → 10 aesthetic scores

All attention and transformer components are implemented from scratch:
no nn.MultiheadAttention, no nn.TransformerEncoder.
"""

import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


# ─── Primitive attention blocks ───────────────────────────────────────────────

class ScaledDotProductAttention(nn.Module):
    """Vanilla scaled dot-product attention."""

    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        Q: torch.Tensor,   # [B, H, N, d_k]
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)  # [B, H, N, N]
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = self.dropout(F.softmax(scores, dim=-1))
        return torch.matmul(attn, V), attn  # ([B,H,N,d_k], [B,H,N,N])


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention built from scratch (no nn.MultiheadAttention)."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)
        self.attn_fn = ScaledDotProductAttention(dropout)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        B, N, D = x.shape
        H, d_k = self.num_heads, self.d_k

        # Project and split into heads: [B, H, N, d_k]
        Q = self.W_q(x).view(B, N, H, d_k).transpose(1, 2)
        K = self.W_k(x).view(B, N, H, d_k).transpose(1, 2)
        V = self.W_v(x).view(B, N, H, d_k).transpose(1, 2)

        out, _ = self.attn_fn(Q, K, V, mask)  # [B, H, N, d_k]
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.W_o(out)


class PositionwiseFeedForward(nn.Module):
    """Two-layer FFN with GELU activation."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.drop(F.gelu(self.fc1(x))))


class TransformerEncoderLayer(nn.Module):
    """
    Pre-norm transformer encoder layer.

    x → LayerNorm → MultiHeadAttention → residual
      → LayerNorm → FFN               → residual
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = x + self.drop(self.attn(self.norm1(x), mask))
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


# ─── Full trainable model ─────────────────────────────────────────────────────

class AestheticTransformer(nn.Module):
    """
    Learns to map CLIP patch tokens → aesthetic match scores.

    Input:  [B, 50, 768]  —  last_hidden_state from CLIP's vision_model
    Output: [B, num_aesthetics]  —  raw logits (apply sigmoid×10 for [0,10] scores)

    ViT-B/32 geometry:
      • Image size 224, patch size 32  →  7×7 = 49 patches  +  1 CLS  =  50 tokens
      • Vision backbone hidden dim: 768 (before the 512-dim linear projection)
    """

    CLIP_DIM  = 768  # ViT-B/32 hidden dim before projection
    N_PATCHES = 50   # 1 CLS + 7×7 patches

    def __init__(
        self,
        num_aesthetics: int = 10,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Project CLIP hidden states down to our working dimension
        self.input_proj = nn.Linear(self.CLIP_DIM, d_model)

        # Learned positional encoding (one vector per sequence position)
        self.pos_emb = nn.Parameter(
            torch.randn(1, self.N_PATCHES, d_model) * 0.02
        )

        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
             for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

        # Classification head: CLS token → aesthetic scores
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_aesthetics),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch_tokens: [B, 50, 768]  CLIP vision_model.last_hidden_state
        Returns:
            logits: [B, num_aesthetics]
        """
        x = self.input_proj(patch_tokens) + self.pos_emb  # [B, 50, d_model]
        for layer in self.layers:
            x = layer(x)
        cls = self.norm(x)[:, 0]   # CLS token after final norm
        return self.head(cls)      # [B, num_aesthetics]


# ─── Inference wrapper ────────────────────────────────────────────────────────

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
    return "Little to no match"


class AestheticScorerV2:
    """
    Drop-in complement to AestheticScorer using the trained transformer.

    Frozen CLIP extracts 50×768 patch tokens; the custom AestheticTransformer
    maps them to 10 aesthetic scores in [0, 10] via sigmoid × 10.

    Parameters
    ----------
    checkpoint_path : path to a .pt file saved by train_transformer.py,
                      or None to use random (untrained) weights.
    aesthetic_names : ordered list of aesthetic keys; read from checkpoint
                      when available.
    device          : "cuda" / "mps" / "cpu"; auto-detected when None.
    """

    def __init__(
        self,
        checkpoint_path: str | None = None,
        aesthetic_names: list[str] | None = None,
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

        print(f"[AestheticScorerV2] Loading CLIP on {device} …")
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip.eval()
        for p in self.clip.parameters():
            p.requires_grad = False

        self.transformer = AestheticTransformer().to(device)

        if checkpoint_path and os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
            self.transformer.load_state_dict(ckpt["model"])
            self.aesthetic_names: list[str] = ckpt.get(
                "aesthetic_names", aesthetic_names or []
            )
            print(f"[AestheticScorerV2] Loaded checkpoint from {checkpoint_path}")
        else:
            self.aesthetic_names = aesthetic_names or []
            if checkpoint_path:
                print(
                    f"[AestheticScorerV2] No checkpoint at {checkpoint_path}; "
                    "using random weights."
                )

        self.transformer.eval()

    # ── Feature extraction ────────────────────────────────────────────────

    def extract_patch_tokens(self, images: list[Image.Image]) -> torch.Tensor:
        """Return [B, 50, 768] patch token tensors from CLIP's vision backbone."""
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            vision_out = self.clip.vision_model(
                pixel_values=inputs["pixel_values"]
            )
        return vision_out.last_hidden_state  # [B, 50, 768]

    # ── Scoring ───────────────────────────────────────────────────────────

    def score_images(self, images: list[Image.Image]) -> np.ndarray:
        """
        Return aesthetic scores in [0, 10] for a batch of images.

        Shape: [N, num_aesthetics].
        """
        patch_tokens = self.extract_patch_tokens(images)
        with torch.no_grad():
            logits = self.transformer(patch_tokens)
        scores = (torch.sigmoid(logits) * 10).cpu().numpy()
        return scores  # [N, 10] in [0, 10]

    def rank_aesthetics(self, image: Image.Image) -> list[dict]:
        """
        Score a single image against all aesthetics.

        Returns a list of dicts sorted by score descending, each with:
            aesthetic – aesthetic key
            score     – 0-10 transformer score (sigmoid-scaled)
            label     – qualitative label string
        """
        scores = self.score_images([image])[0]  # [num_aesthetics]
        names = self.aesthetic_names or [str(i) for i in range(len(scores))]
        results = [
            {
                "aesthetic": name,
                "score": round(float(scores[i]), 2),
                "label": _score_label(float(scores[i])),
            }
            for i, name in enumerate(names)
        ]
        return sorted(results, key=lambda r: r["score"], reverse=True)

    def score_image(self, image: Image.Image, aesthetic_name: str) -> dict:
        """Rate an outfit image against a specific aesthetic."""
        ranked = self.rank_aesthetics(image)
        for r in ranked:
            if r["aesthetic"] == aesthetic_name:
                return r
        raise ValueError(f"Aesthetic '{aesthetic_name}' not in {self.aesthetic_names}")
