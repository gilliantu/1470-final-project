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
import torchvision.transforms as T


# ── Background removal ────────────────────────────────────────────────────────

def remove_background(img: Image.Image) -> Image.Image:
    """
    Strip the background from an image, leaving only the foreground subject
    (person / outfit) composited onto a plain white background.

    Uses rembg (U2-Net) for segmentation.  Falls back to the original image
    if rembg is not installed.
    """
    try:
        from rembg import remove
        rgba = remove(img.convert("RGBA"))          # RGBA: alpha = subject mask
        white = Image.new("RGB", rgba.size, (255, 255, 255))
        white.paste(rgba, mask=rgba.split()[3])     # paste subject over white
        return white
    except ImportError:
        return img


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

        Background is removed from each image first so CLIP attends to the
        outfit / person rather than the surrounding scene.

        Returns:
          patch_tokens: [B, 50, 768]  raw vision backbone hidden states
          clip_emb:     [B, 512]      L2-normalised projected embedding
                                      (same feature used by the prototype scorer)
        """
        images = [remove_background(img) for img in images]
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


# ── Patch embedder (replaces CLIP's frozen ViT backbone) ─────────────────────

class PatchEmbedder(nn.Module):
    """
    ViT-style patch tokenizer implemented from scratch — no CLIP dependency.

    A single Conv2d with kernel_size=patch_size, stride=patch_size is
    mathematically equivalent to flattening each patch and multiplying by a
    shared linear projection.  This is exactly what CLIP ViT-B/32 does
    internally, but here the weights are randomly initialised and trained
    end-to-end.

    For 224×224 input with patch_size=32: 7×7 = 49 spatial patches.
    Prepending a learnable CLS token gives 50 tokens — matching CLIP ViT-B/32
    so the downstream transformer and positional encoding are unchanged.
    """

    IMG_SIZE   = 224
    PATCH_SIZE = 32

    def __init__(self, patch_dim: int = 768, in_channels: int = 3) -> None:
        super().__init__()
        self.patch_proj = nn.Conv2d(
            in_channels, patch_dim,
            kernel_size=self.PATCH_SIZE, stride=self.PATCH_SIZE, bias=False,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, patch_dim))
        nn.init.trunc_normal_(self.cls_token,      std=0.02)
        nn.init.trunc_normal_(self.patch_proj.weight, std=0.02)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        pixel_values: [B, 3, 224, 224]
        Returns:      [B, 50, patch_dim]  (50 = 1 CLS + 49 patches)
        """
        B = pixel_values.size(0)
        x = self.patch_proj(pixel_values)   # [B, patch_dim, 7, 7]
        x = x.flatten(2).transpose(1, 2)   # [B, 49, patch_dim]
        cls = self.cls_token.expand(B, -1, -1)
        return torch.cat([cls, x], dim=1)  # [B, 50, patch_dim]


# ── Pretrained patch embedder (ImageNet ResNet-50 backbone) ──────────────────

class PretrainedPatchEmbedder(nn.Module):
    """
    Multi-scale ResNet-50 patch embedder (ImageNet weights, no CLIP).

    Fuses layer3 [B, 1024, 14, 14] and layer4 [B, 2048, 7, 7] features:
      - layer3 pooled to 7×7 captures mid-level textures and fabric patterns
      - layer4 at 7×7 captures high-level semantic style cues
      - concatenated [B, 3072, 7, 7] → 1×1 conv → [B, patch_dim, 7, 7]
      - reshape + CLS → [B, 50, patch_dim]

    Using both scales gives the transformer richer style information than
    layer4 alone — fabric texture comes from layer3, overall outfit structure
    from layer4.

    Backbone is frozen by default (fast, cacheable).
    Set fine_tune=True to also train layer3 + layer4 (more accurate, slower).
    """

    def __init__(self, patch_dim: int = 768, fine_tune: bool = False) -> None:
        super().__init__()
        import torchvision.models as tvm
        backbone = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V1)

        self.early = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2,
        )
        self.layer3   = backbone.layer3          # [B, 1024, 14, 14]
        self.layer4   = backbone.layer4          # [B, 2048,  7,  7]
        self.pool3    = nn.AdaptiveAvgPool2d(7)  # [B, 1024,  7,  7]

        # Freeze early layers always; optionally unfreeze layer3+layer4
        for p in self.early.parameters():
            p.requires_grad = False
        for p in self.layer3.parameters():
            p.requires_grad = fine_tune
        for p in self.layer4.parameters():
            p.requires_grad = fine_tune

        self.proj      = nn.Conv2d(1024 + 2048, patch_dim, kernel_size=1)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, patch_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        pixel_values: [B, 3, 224, 224]
        Returns:      [B, 50, patch_dim]
        """
        B  = pixel_values.size(0)
        x  = self.early(pixel_values)           # [B, 512,  28, 28]
        l3 = self.layer3(x)                     # [B, 1024, 14, 14]
        l4 = self.layer4(l3)                    # [B, 2048,  7,  7]
        l3p = self.pool3(l3)                    # [B, 1024,  7,  7]
        fused = torch.cat([l3p, l4], dim=1)     # [B, 3072,  7,  7]
        x  = self.proj(fused)                   # [B, patch_dim, 7, 7]
        x  = x.flatten(2).transpose(1, 2)       # [B, 49, patch_dim]
        cls = self.cls_token.expand(B, -1, -1)
        return torch.cat([cls, x], dim=1)       # [B, 50, patch_dim]


# ── Aesthetic transformer V2 (no CLIP embedding bridge) ──────────────────────

class AestheticTransformerV2(nn.Module):
    """
    Encoder-only aesthetic classifier with zero CLIP dependency.

    Differences from AestheticTransformer:
      - No clip_bridge: the CLS token alone summarises the image
      - input_proj maps patch_dim → d_model (patch_dim comes from PatchEmbedder)
      - All attention blocks, positional encoding, and the MLP head are
        identical to AestheticTransformer
    """

    N_PATCHES = 50  # 1 CLS + 7×7 for 224×224 / patch_size=32

    def __init__(
        self,
        num_aesthetics: int = 10,
        patch_dim: int = 768,
        d_model: int = 384,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(patch_dim, d_model)

        pe = positional_encoding(self.N_PATCHES, d_model)
        self.register_buffer("pos_enc", pe.unsqueeze(0))  # [1, 50, d_model]

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
        x = self.input_proj(patch_tokens) + self.pos_enc  # [B, 50, d_model]
        for block in self.blocks:
            x = block(x)
        cls = self.norm(x)[:, 0]   # CLS token  [B, d_model]
        return self.head(cls)       # [B, num_aesthetics]


# ── Fully CLIP-free inference wrapper ────────────────────────────────────────

class AestheticScorerV3:
    """
    Fully CLIP-free aesthetic scorer.

    PretrainedPatchEmbedder (ResNet-50, ImageNet weights) replaces CLIP's ViT
    backbone, giving semantically rich patch tokens without any CLIP dependency.
    AestheticTransformerV2 classifies from the CLS token.

    Image preprocessing uses torchvision transforms (replaces CLIPProcessor).
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
        """Background-remove then PatchEmbed → [B, 50, patch_dim]."""
        images = [remove_background(img) for img in images]
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
