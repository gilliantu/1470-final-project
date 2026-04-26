"""
transformer_model.py — Aesthetic transformer built on student_hw_model_transformer.py.

Starting from the homework attention and block structure, improved with
HuggingFace-style conventions:

  AttentionMatrix     — kept; added attention dropout
  AttentionHead       — kept; removed is_self_attention (encoder-only, no causal mask)
  MultiHeadedAttention— configurable num_heads (was hardcoded 3); nn.ModuleList for heads
  TransformerBlock    — encoder-only (removed cross-attention); pre-LayerNorm;
                        GELU instead of ReLU; separate residual dropout
  positional_encoding — kept exactly from student code
  AestheticTransformer— new: projects CLIP tokens → d_model, stacks encoder blocks,
                        classifies from CLS token

CLIP is used only in AestheticScorerV2.extract_patch_tokens (frozen, no gradients).
The transformer itself has zero dependency on CLIP.
"""

import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


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
      - Pre-LayerNorm (norm before sublayer, not after) — HuggingFace ViT/GPT-2 style,
        more stable gradients in deep stacks
      - GELU activation instead of ReLU — HuggingFace standard
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
        # Pre-norm self-attention + residual
        normed = self.norm1(x)
        x = x + self.dropout1(self.attention(normed, normed, normed))
        # Pre-norm FFN + residual
        x = x + self.dropout2(self.feed_forward(self.norm2(x)))
        return x


# ── Positional encoding (kept exactly from student_hw_model_transformer.py) ──

def positional_encoding(length: int, depth: int) -> torch.Tensor:
    """
    Sinusoidal positional encoding (from student_hw_model_transformer.py).

    :param length: number of positions (sequence length)
    :param depth:  embedding dimension
    :return:       torch.FloatTensor [length x depth]
    """
    depth     = depth // 2
    positions = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    depths    = torch.arange(depth,  dtype=torch.float32).unsqueeze(0) / depth
    angle_rads = positions / (10000 ** depths)
    return torch.cat([torch.sin(angle_rads), torch.cos(angle_rads)], dim=1)


# ── Full aesthetic transformer ────────────────────────────────────────────────

class AestheticTransformer(nn.Module):
    """
    Encoder transformer for aesthetic classification.

    Inputs:
      patch_tokens [B, 50, 768]  — raw CLIP vision backbone patch tokens
      clip_emb     [B, 512]      — L2-normalised CLIP projected embedding
                                   (same feature the prototype scorer uses)
    Output: logits [B, num_aesthetics]
            Apply z-score normalisation at inference to get [0, 10] scores.

    Architecture:
      1. Linear projection 768 → d_model
      2. Sinusoidal positional encoding (fixed buffer, from student code)
      3. num_layers × TransformerBlock  (pre-norm, GELU, configurable heads)
      4. Final LayerNorm on CLS token
      5. clip_bridge fuses the projected CLIP embedding into the CLS output
         so the model sees both spatial patch detail AND the global semantic
         signal already aligned with the text prototype space
      6. Two-layer MLP head → aesthetic logits
    """

    CLIP_DIM   = 768
    CLIP_PROJ  = 512   # projected CLIP embedding dimension
    N_PATCHES  = 50    # 1 CLS + 7×7 patches for ViT-B/32

    def __init__(
        self,
        num_aesthetics: int = 10,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.input_proj  = nn.Linear(self.CLIP_DIM,  d_model)
        # Projects the 512-dim CLIP global embedding into d_model space
        # and adds it to the CLS token — gives the model direct access to
        # the same feature the original prototype scorer uses
        self.clip_bridge = nn.Linear(self.CLIP_PROJ, d_model, bias=False)

        pe = positional_encoding(self.N_PATCHES, d_model)
        self.register_buffer("pos_enc", pe.unsqueeze(0))   # [1, 50, d_model]

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

    def forward(
        self,
        patch_tokens: torch.Tensor,
        clip_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        patch_tokens: [B, 50, 768]
        clip_emb:     [B, 512]  L2-normalised projected CLIP embedding
        Returns logits: [B, num_aesthetics]
        """
        x = self.input_proj(patch_tokens) + self.pos_enc   # [B, 50, d_model]
        for block in self.blocks:
            x = block(x)
        cls = self.norm(x)[:, 0]                           # [B, d_model]
        # Fuse global CLIP embedding — residual addition so the transformer
        # can refine the CLIP signal rather than learn it from scratch
        cls = cls + self.clip_bridge(clip_emb)
        return self.head(cls)                              # [B, num_aesthetics]


# ── Inference wrapper ─────────────────────────────────────────────────────────

def _score_label(score: float) -> str:
    if score >= 8.5: return "Perfect match"
    if score >= 7.0: return "Strong match"
    if score >= 5.5: return "Good match"
    if score >= 4.0: return "Moderate match"
    if score >= 2.5: return "Weak match"
    return "Little to no match"


class AestheticScorerV2:
    """
    Frozen CLIP (feature extraction only) + trained AestheticTransformer.

    CLIP is accessed exclusively in extract_patch_tokens.
    The transformer has zero CLIP dependency.
    """

    def __init__(
        self,
        checkpoint_path: str | None = None,
        aesthetic_names: list[str] | None = None,
        device: str | None = None,
    ) -> None:
        if device is None:
            if torch.cuda.is_available():           device = "cuda"
            elif torch.backends.mps.is_available(): device = "mps"
            else:                                   device = "cpu"
        self.device = device

        print(f"[AestheticScorerV2] Loading CLIP on {device} …")
        self.clip      = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip.eval()
        for p in self.clip.parameters():
            p.requires_grad = False

        self.transformer = AestheticTransformer().to(device)

        if checkpoint_path and os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
            self.transformer.load_state_dict(ckpt["model"])
            self.aesthetic_names: list[str] = ckpt.get("aesthetic_names", aesthetic_names or [])
            print(f"[AestheticScorerV2] Checkpoint loaded from {checkpoint_path}")
        else:
            self.aesthetic_names = aesthetic_names or []
            if checkpoint_path:
                print(f"[AestheticScorerV2] No checkpoint at {checkpoint_path}; random weights.")

        self.transformer.eval()

    # ── Feature extraction (only CLIP usage) ─────────────────────────────

    def extract_features(
        self, images: list[Image.Image]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extract both feature types from frozen CLIP in one forward pass.

        Returns:
          patch_tokens: [B, 50, 768]  raw vision backbone hidden states
          clip_emb:     [B, 512]      L2-normalised projected embedding
                                      (same feature used by the prototype scorer)
        """
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out      = self.clip.vision_model(pixel_values=inputs["pixel_values"])
            clip_emb = self.clip.visual_projection(out.pooler_output)   # [B, 512]
            clip_emb = clip_emb / clip_emb.norm(dim=-1, keepdim=True)  # L2 normalise
        return out.last_hidden_state, clip_emb

    # kept for backward-compat with train_transformer.py cache extraction
    def extract_patch_tokens(self, images: list[Image.Image]) -> torch.Tensor:
        return self.extract_features(images)[0]

    # ── Scoring ───────────────────────────────────────────────────────────

    @staticmethod
    def _zscore(raw: np.ndarray) -> np.ndarray:
        """
        Within-image z-score normalisation → [0, 10].
        Identical to the original AestheticScorer._zscore_scores so rankings
        are directly comparable.
        """
        mean = raw.mean(axis=-1, keepdims=True)
        std  = raw.std(axis=-1,  keepdims=True)
        std  = np.where(std < 1e-6, 1.0, std)
        return np.clip(5.0 + (raw - mean) / std * 2.0, 0.0, 10.0)

    @torch.no_grad()
    def score_images(self, images: list[Image.Image]) -> np.ndarray:
        """Return aesthetic scores in [0, 10], shape [N, num_aesthetics]."""
        tokens, clip_emb = self.extract_features(images)
        self.transformer.eval()
        logits = self.transformer(tokens, clip_emb).cpu().numpy()
        return self._zscore(logits)

    def rank_aesthetics(self, image: Image.Image) -> list[dict]:
        """Score one image; returns list sorted by score descending."""
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
