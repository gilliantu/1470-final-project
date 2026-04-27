"""
transformer_model.py — CLIP-free aesthetic transformer (V3).

Components
----------
  AttentionMatrix / AttentionHead / MultiHeadedAttention / TransformerBlock
      Encoder-only attention stack adapted from student homework code.

  PretrainedPatchEmbedder
      ResNet-50 multi-scale patch extractor (ImageNet weights).
      Fuses layer3 + layer4 → 1×1 conv → LayerNorm → [B, 50, patch_dim].

  AestheticTransformerV2
      Encoder-only transformer classifier over patch tokens.
      No positional encoding (CNN patch tokens have no inherent order).

  AestheticScorerV3
      Inference wrapper: embedder + transformer + z-score normalisation.
"""

import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T


# ── Attention (from student_hw_model_transformer.py) ─────────────────────────

class AttentionMatrix(nn.Module):
    """
    Scaled dot-product attention weights.

    Change from student code: added attention dropout (HuggingFace standard);
    removed causal mask — encoder-only classification needs no autoregressive mask.
    """

    def __init__(self, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, K: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
        """
        K: [B x seq_keys    x d_k]
        Q: [B x seq_queries x d_k]
        Returns attention weights [B x seq_queries x seq_keys]
        """
        d_k    = K.size(-1)
        scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(d_k)
        return self.dropout(F.softmax(scores, dim=-1))


class AttentionHead(nn.Module):
    """
    Single attention head.

    Change from student code: removed is_self_attention / use_mask parameter
    since the encoder never needs a causal mask.
    """

    def __init__(self, input_size: int, output_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.key      = nn.Linear(input_size, output_size, bias=False)
        self.value    = nn.Linear(input_size, output_size, bias=False)
        self.query    = nn.Linear(input_size, output_size, bias=False)
        self.attention = AttentionMatrix(dropout)

    def forward(
        self,
        inputs_for_keys:    torch.Tensor,
        inputs_for_values:  torch.Tensor,
        inputs_for_queries: torch.Tensor,
    ) -> torch.Tensor:
        """
        inputs: [B x seq x input_size]
        Returns [B x seq_queries x output_size]
        """
        K = self.key(inputs_for_keys)
        V = self.value(inputs_for_values)
        Q = self.query(inputs_for_queries)
        return torch.bmm(self.attention(K, Q), V)


class MultiHeadedAttention(nn.Module):
    """
    Multi-head attention with configurable number of heads.

    Changes from student code:
      - num_heads is a parameter (was hardcoded to 3)
      - Uses nn.ModuleList so all heads are tracked by the optimizer
      - Output projection renamed to out_proj (was linear)
    """

    def __init__(self, emb_sz: int, num_heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        assert emb_sz % num_heads == 0, "emb_sz must be divisible by num_heads"
        head_dim   = emb_sz // num_heads
        self.heads = nn.ModuleList([
            AttentionHead(emb_sz, head_dim, dropout) for _ in range(num_heads)
        ])
        self.out_proj = nn.Linear(emb_sz, emb_sz)
        self.dropout  = nn.Dropout(dropout)

    def forward(
        self,
        inputs_for_keys:    torch.Tensor,
        inputs_for_values:  torch.Tensor,
        inputs_for_queries: torch.Tensor,
    ) -> torch.Tensor:
        """Returns [B x seq_queries x emb_sz]."""
        combined = torch.cat(
            [h(inputs_for_keys, inputs_for_values, inputs_for_queries) for h in self.heads],
            dim=-1,
        )
        return self.dropout(self.out_proj(combined))


# ── Transformer block (from student code, improved) ──────────────────────────

class TransformerBlock(nn.Module):
    """
    Encoder-only transformer block.

    Changes from student code:
      - Removed cross-attention sublayer — encoders for classification don't need it
      - Pre-LayerNorm (norm before sublayer, not after) — more stable gradients
      - GELU activation instead of ReLU
      - FFN width is 4× emb_sz (standard transformer ratio)
    """

    def __init__(self, emb_sz: int, num_heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1     = nn.LayerNorm(emb_sz)
        self.norm2     = nn.LayerNorm(emb_sz)
        self.attention = MultiHeadedAttention(emb_sz, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(emb_sz, emb_sz * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(emb_sz * 4, emb_sz),
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B x seq_len x emb_sz]"""
        normed = self.norm1(x)
        x = x + self.dropout1(self.attention(normed, normed, normed))
        x = x + self.dropout2(self.feed_forward(self.norm2(x)))
        return x


# ── CLIP-free image preprocessing ────────────────────────────────────────────

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)

def build_transform(img_size: int = 224) -> T.Compose:
    """Val / inference transform — deterministic."""
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


def build_train_transform(img_size: int = 224) -> T.Compose:
    """Training transform — adds augmentation to reduce overfitting."""
    return T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.75, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        T.ToTensor(),
        T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


# ── Pretrained patch embedder (ImageNet ResNet-50 backbone) ──────────────────

class PretrainedPatchEmbedder(nn.Module):
    """
    Multi-scale ResNet-50 patch embedder (ImageNet weights, no CLIP).

    Fuses layer3 [B, 1024, 14, 14] and layer4 [B, 2048, 7, 7] features:
      - layer3 pooled to 7×7: mid-level textures and fabric patterns
      - layer4 at 7×7: high-level semantic style cues
      - concatenated [B, 3072, 7, 7] → 1×1 conv → LayerNorm → [B, patch_dim, 7, 7]
      - reshape + CLS → [B, 50, patch_dim]

    LayerNorm after the projection normalises patch features to a consistent
    scale before the transformer, avoiding scale mismatch with the CLS token.

    Backbone is frozen by default (fast, cacheable).
    Set fine_tune=True to also train layer3 + layer4.
    """

    def __init__(self, patch_dim: int = 768, fine_tune: bool = False) -> None:
        super().__init__()
        import torchvision.models as tvm
        backbone = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V1)

        self.early = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2,
        )
        self.layer3 = backbone.layer3          # [B, 1024, 14, 14]
        self.layer4 = backbone.layer4          # [B, 2048,  7,  7]
        self.pool3  = nn.AdaptiveAvgPool2d(7)  # [B, 1024,  7,  7]

        for p in self.early.parameters():
            p.requires_grad = False
        for p in self.layer3.parameters():
            p.requires_grad = fine_tune
        for p in self.layer4.parameters():
            p.requires_grad = fine_tune

        self.proj      = nn.Conv2d(1024 + 2048, patch_dim, kernel_size=1)
        self.proj_norm = nn.LayerNorm(patch_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, patch_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        pixel_values: [B, 3, 224, 224]
        Returns:      [B, 50, patch_dim]
        """
        B   = pixel_values.size(0)
        x   = self.early(pixel_values)          # [B, 512,  28, 28]
        l3  = self.layer3(x)                    # [B, 1024, 14, 14]
        l4  = self.layer4(l3)                   # [B, 2048,  7,  7]
        l3p = self.pool3(l3)                    # [B, 1024,  7,  7]
        fused = torch.cat([l3p, l4], dim=1)     # [B, 3072,  7,  7]
        x   = self.proj(fused)                  # [B, patch_dim, 7, 7]
        x   = x.flatten(2).transpose(1, 2)      # [B, 49, patch_dim]
        x   = self.proj_norm(x)                 # normalise patch features
        cls = self.cls_token.expand(B, -1, -1)
        return torch.cat([cls, x], dim=1)       # [B, 50, patch_dim]


# ── Aesthetic transformer V2 (CLIP-free) ─────────────────────────────────────

class AestheticTransformerV2(nn.Module):
    """
    Encoder-only aesthetic classifier with zero CLIP dependency.

    No positional encoding: CNN patch tokens carry spatial information in their
    feature values (not token order), so absolute PE adds noise.
    Dropout raised to 0.2 for regularisation on small training sets.
    """

    N_PATCHES = 50  # 1 CLS + 7×7 for 224×224 / patch_size=32

    def __init__(
        self,
        num_aesthetics: int = 10,
        patch_dim: int = 768,
        d_model: int = 384,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(patch_dim, d_model)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_aesthetics),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """
        patch_tokens: [B, 50, patch_dim]
        Returns logits: [B, num_aesthetics]
        """
        x = self.input_proj(patch_tokens)  # [B, 50, d_model]
        for block in self.blocks:
            x = block(x)
        cls = self.norm(x)[:, 0]           # CLS token  [B, d_model]
        return self.head(cls)              # [B, num_aesthetics]


# ── Inference wrapper ─────────────────────────────────────────────────────────

def _score_label(score: float) -> str:
    if score >= 8.5: return "Perfect match"
    if score >= 7.0: return "Strong match"
    if score >= 5.5: return "Good match"
    if score >= 4.0: return "Moderate match"
    if score >= 2.5: return "Weak match"
    return "Little to no match"


class AestheticScorerV3:
    """
    Fully CLIP-free aesthetic scorer.

    PretrainedPatchEmbedder (ResNet-50, ImageNet weights) extracts multi-scale
    patch tokens. AestheticTransformerV2 classifies from the CLS token.
    Scores are z-normalised per image to the 0–10 range at inference.
    """

    def __init__(
        self,
        checkpoint_path: str | None = None,
        aesthetic_names: list[str] | None = None,
        patch_dim: int = 768,
        d_model: int = 384,
        fine_tune: bool = False,
        device: str | None = None,
    ) -> None:
        if device is None:
            if torch.cuda.is_available():           device = "cuda"
            elif torch.backends.mps.is_available(): device = "mps"
            else:                                   device = "cpu"
        self.device = device

        self.embedder    = PretrainedPatchEmbedder(patch_dim=patch_dim, fine_tune=fine_tune).to(device)
        self.transformer = AestheticTransformerV2(patch_dim=patch_dim, d_model=d_model).to(device)
        self.transform   = build_transform()

        if checkpoint_path and os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
            self.embedder.load_state_dict(ckpt["embedder"])
            self.transformer.load_state_dict(ckpt["transformer"])
            self.aesthetic_names: list[str] = ckpt.get("aesthetic_names", aesthetic_names or [])
            print(f"[AestheticScorerV3] Checkpoint loaded from {checkpoint_path}")
        else:
            self.aesthetic_names = aesthetic_names or []
            if checkpoint_path:
                print(f"[AestheticScorerV3] No checkpoint at {checkpoint_path}; random weights.")

        self.embedder.eval()
        self.transformer.eval()

    def preprocess(self, images: list[Image.Image]) -> torch.Tensor:
        """PIL images → normalised [B, 3, 224, 224] tensor."""
        return torch.stack(
            [self.transform(img.convert("RGB")) for img in images]
        ).to(self.device)

    def extract_features(self, images: list[Image.Image]) -> torch.Tensor:
        """PatchEmbed images → [B, 50, patch_dim]."""
        pixel_values = self.preprocess(images)
        with torch.no_grad():
            return self.embedder(pixel_values)

    @staticmethod
    def _zscore(raw: np.ndarray) -> np.ndarray:
        mean = raw.mean(axis=-1, keepdims=True)
        std  = raw.std(axis=-1,  keepdims=True)
        std  = np.where(std < 1e-6, 1.0, std)
        return np.clip(5.0 + (raw - mean) / std * 2.0, 0.0, 10.0)

    @torch.no_grad()
    def score_images(self, images: list[Image.Image]) -> np.ndarray:
        """Return aesthetic scores in [0, 10], shape [N, num_aesthetics]."""
        tokens = self.extract_features(images)
        self.transformer.eval()
        logits = self.transformer(tokens).cpu().numpy()
        return self._zscore(logits)

    def rank_aesthetics(self, image: Image.Image) -> list[dict]:
        scores = self.score_images([image])[0]
        names  = self.aesthetic_names or [str(i) for i in range(len(scores))]
        results = [
            {"aesthetic": name,
             "score":     round(float(scores[i]), 2),
             "label":     _score_label(float(scores[i]))}
            for i, name in enumerate(names)
        ]
        return sorted(results, key=lambda r: r["score"], reverse=True)

    def score_image(self, image: Image.Image, aesthetic_name: str) -> dict:
        for r in self.rank_aesthetics(image):
            if r["aesthetic"] == aesthetic_name:
                return r
        raise ValueError(f"'{aesthetic_name}' not in {self.aesthetic_names}")
