"""
transformer_model.py — Custom transformer for aesthetic scoring.

All transformer components (attention, FFN, layer norm) are implemented
from scratch in NumPy — no PyTorch, no Keras.
PyTorch is used only to load the frozen CLIP feature extractor.

Architecture
------------
1. CLIP ViT-B/32 (frozen PyTorch) → 50 patch tokens × 768-dim
2. NumPy transformer:
   a. Linear projection 768 → d_model
   b. Learned positional encoding
   c. N × TransformerEncoderLayer (multi-head self-attention + FFN)
   d. CLS token → MLP head → 10 aesthetic scores
3. Training via manual backprop + Adam (all NumPy)
"""

import os
import math
import numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
import torch


# ── Activations ───────────────────────────────────────────────────────────────

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))

def gelu_grad(x):
    c = np.sqrt(2.0 / np.pi)
    t = np.tanh(c * (x + 0.044715 * x ** 3))
    return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t ** 2) * c * (1.0 + 3 * 0.044715 * x ** 2)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


# ── Positional Encoding ───────────────────────────────────────────────────────

def sinusoidal_encoding(n_positions: int, d_model: int) -> np.ndarray:
    """
    Sinusoidal positional encoding (Vaswani et al. 2017).

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Returns float32 [n_positions, d_model] with values in [-1, 1].
    """
    pos   = np.arange(n_positions, dtype=np.float32)[:, None]   # [N, 1]
    dims  = np.arange(d_model,     dtype=np.float32)[None, :]   # [1, D]
    freqs = np.power(10000.0, (2 * (dims // 2)) / d_model)
    angles = pos / freqs                                         # [N, D]
    enc = np.where(dims % 2 == 0, np.sin(angles), np.cos(angles))
    return enc.astype(np.float32)


# ── Layer Normalization ───────────────────────────────────────────────────────

def layer_norm_fwd(x, gamma, beta, eps=1e-5):
    """x: [..., D]. Returns (y, cache)."""
    mean  = np.mean(x, axis=-1, keepdims=True)
    var   = np.var(x,  axis=-1, keepdims=True)
    x_hat = (x - mean) / np.sqrt(var + eps)
    return gamma * x_hat + beta, (x_hat, var, gamma, eps)

def layer_norm_bwd(dout, cache):
    """Returns (dx, dgamma, dbeta)."""
    x_hat, var, gamma, eps = cache
    N = x_hat.shape[-1]
    sum_axes = tuple(range(dout.ndim - 1))
    dgamma = np.sum(dout * x_hat, axis=sum_axes)
    dbeta  = np.sum(dout,         axis=sum_axes)
    dx_hat = dout * gamma
    std_inv = 1.0 / np.sqrt(var + eps)
    dx = std_inv * (
        dx_hat
        - np.mean(dx_hat,          axis=-1, keepdims=True)
        - x_hat * np.mean(dx_hat * x_hat, axis=-1, keepdims=True)
    )
    return dx, dgamma, dbeta


# ── Scaled Dot-Product Attention ──────────────────────────────────────────────

def sdp_attn_fwd(Q, K, V):
    """Q/K/V: [B, H, N, d_k]. Returns (out, cache)."""
    scale  = math.sqrt(Q.shape[-1])
    scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / scale   # [B,H,N,N]
    attn   = softmax(scores, axis=-1)
    out    = np.matmul(attn, V)                                # [B,H,N,d_k]
    return out, (Q, K, V, attn, scale)

def sdp_attn_bwd(dout, cache):
    """Returns (dQ, dK, dV)."""
    Q, K, V, attn, scale = cache
    dV      = np.matmul(attn.transpose(0, 1, 3, 2), dout)
    d_attn  = np.matmul(dout, V.transpose(0, 1, 3, 2))
    d_scores = attn * (d_attn - np.sum(d_attn * attn, axis=-1, keepdims=True))
    d_scores /= scale
    dQ = np.matmul(d_scores,                       K)
    dK = np.matmul(d_scores.transpose(0, 1, 3, 2), Q)
    return dQ, dK, dV


# ── Multi-Head Attention ──────────────────────────────────────────────────────

def mha_fwd(x, W_q, W_k, W_v, W_o, num_heads):
    """x: [B, N, D]. Returns (out, cache)."""
    B, N, D = x.shape
    d_k = D // num_heads
    Q = (x @ W_q).reshape(B, N, num_heads, d_k).transpose(0, 2, 1, 3)
    K = (x @ W_k).reshape(B, N, num_heads, d_k).transpose(0, 2, 1, 3)
    V = (x @ W_v).reshape(B, N, num_heads, d_k).transpose(0, 2, 1, 3)
    attn_out, sdp_cache = sdp_attn_fwd(Q, K, V)
    merged = attn_out.transpose(0, 2, 1, 3).reshape(B, N, D)
    out    = merged @ W_o
    return out, (x, merged, sdp_cache, W_q, W_k, W_v, W_o, num_heads)

def mha_bwd(dout, cache):
    """Returns (dx, dW_q, dW_k, dW_v, dW_o)."""
    x, merged, sdp_cache, W_q, W_k, W_v, W_o, num_heads = cache
    B, N, D = x.shape
    d_k = D // num_heads
    dW_o    = merged.reshape(-1, D).T @ dout.reshape(-1, D)
    d_merged = dout @ W_o.T
    d_attn_out = d_merged.reshape(B, N, num_heads, d_k).transpose(0, 2, 1, 3)
    dQ_h, dK_h, dV_h = sdp_attn_bwd(d_attn_out, sdp_cache)
    dQ = dQ_h.transpose(0, 2, 1, 3).reshape(B, N, D)
    dK = dK_h.transpose(0, 2, 1, 3).reshape(B, N, D)
    dV = dV_h.transpose(0, 2, 1, 3).reshape(B, N, D)
    dW_q = x.reshape(-1, D).T @ dQ.reshape(-1, D)
    dW_k = x.reshape(-1, D).T @ dK.reshape(-1, D)
    dW_v = x.reshape(-1, D).T @ dV.reshape(-1, D)
    dx   = dQ @ W_q.T + dK @ W_k.T + dV @ W_v.T
    return dx, dW_q, dW_k, dW_v, dW_o


# ── Feed-Forward Network ──────────────────────────────────────────────────────

def ff_fwd(x, W1, b1, W2, b2):
    """x: [B, N, D] → [B, N, D]."""
    h     = x @ W1 + b1
    h_act = gelu(h)
    out   = h_act @ W2 + b2
    return out, (x, h, h_act, W1, W2)

def ff_bwd(dout, cache):
    """Returns (dx, dW1, db1, dW2, db2)."""
    x, h, h_act, W1, W2 = cache
    dW2   = h_act.reshape(-1, h_act.shape[-1]).T @ dout.reshape(-1, dout.shape[-1])
    db2   = dout.sum(axis=tuple(range(dout.ndim - 1)))
    dh_act = dout @ W2.T
    dh    = dh_act * gelu_grad(h)
    dW1   = x.reshape(-1, x.shape[-1]).T @ dh.reshape(-1, h.shape[-1])
    db1   = dh.sum(axis=tuple(range(dh.ndim - 1)))
    dx    = dh @ W1.T
    return dx, dW1, db1, dW2, db2


# ── Transformer Encoder Layer ─────────────────────────────────────────────────

def encoder_layer_fwd(x, p, num_heads):
    """
    Pre-norm: x → LN → MHA → residual → LN → FFN → residual.
    p: dict with norm/attention/FFN weight keys.
    Returns (x2, cache).
    """
    # Sublayer 1: self-attention
    xn1, ln1_cache = layer_norm_fwd(x, p['norm1_gamma'], p['norm1_beta'])
    attn_out, mha_cache = mha_fwd(xn1, p['W_q'], p['W_k'], p['W_v'], p['W_o'], num_heads)
    x1 = x + attn_out

    # Sublayer 2: FFN
    xn2, ln2_cache = layer_norm_fwd(x1, p['norm2_gamma'], p['norm2_beta'])
    ff_out, ff_cache = ff_fwd(xn2, p['W1'], p['b1'], p['W2'], p['b2'])
    x2 = x1 + ff_out

    return x2, (ln1_cache, mha_cache, ln2_cache, ff_cache)

def encoder_layer_bwd(dout, cache):
    """Returns (dx, grads_dict)."""
    ln1_cache, mha_cache, ln2_cache, ff_cache = cache

    # FFN residual: x2 = x1 + ff_out
    d_x1      = dout.copy()
    d_ff_out  = dout
    d_xn2, dW1, db1, dW2, db2 = ff_bwd(d_ff_out, ff_cache)
    d_x1_ln2, dg2, dbeta2     = layer_norm_bwd(d_xn2, ln2_cache)
    d_x1 += d_x1_ln2

    # Attention residual: x1 = x + attn_out
    dx          = d_x1.copy()
    d_attn_out  = d_x1
    d_xn1, d_Wq, d_Wk, d_Wv, d_Wo = mha_bwd(d_attn_out, mha_cache)
    d_x_ln1, dg1, dbeta1           = layer_norm_bwd(d_xn1, ln1_cache)
    dx += d_x_ln1

    grads = {
        'norm1_gamma': dg1,   'norm1_beta': dbeta1,
        'norm2_gamma': dg2,   'norm2_beta': dbeta2,
        'W_q': d_Wq, 'W_k': d_Wk, 'W_v': d_Wv, 'W_o': d_Wo,
        'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2,
    }
    return dx, grads


# ── Aesthetic Transformer (NumPy) ─────────────────────────────────────────────

class AestheticTransformer:
    """
    Transformer for aesthetic scoring — implemented entirely in NumPy.

    Input:  patch_tokens [B, 50, 768]  (from frozen CLIP vision backbone)
    Output: logits [B, num_aesthetics]  (apply sigmoid×10 for [0,10] scores)

    Defaults: 6 encoder layers, d_model=256, 8 heads, d_ff=1024 (4× d_model).
    Positional encoding: sinusoidal (no CLIP dependency).
    """

    CLIP_DIM  = 768
    N_PATCHES = 50

    def __init__(
        self,
        num_aesthetics: int = 10,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 1024,
        seed: int = 0,
    ):
        self.num_aesthetics = num_aesthetics
        self.d_model        = d_model
        self.num_heads      = num_heads
        self.num_layers     = num_layers
        self.d_ff           = d_ff
        self._rng           = np.random.default_rng(seed)

        self.params = self._init_params()
        self.m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.t = 0

    # ── Parameter initialisation ──────────────────────────────────────────

    def _xavier(self, fan_in, fan_out):
        std = math.sqrt(2.0 / (fan_in + fan_out))
        return self._rng.normal(0.0, std, (fan_in, fan_out)).astype(np.float32)

    def _init_params(self):
        d, dff, H = self.d_model, self.d_ff, self.CLIP_DIM
        p = {}
        p['proj_W']  = self._xavier(H, d)
        p['proj_b']  = np.zeros(d, dtype=np.float32)
        enc = sinusoidal_encoding(self.N_PATCHES, d)               # [50, d]
        p['pos_emb'] = (enc / (enc.std() + 1e-8) * 0.02)[None].astype(np.float32)
        for i in range(self.num_layers):
            pf = f'l{i}_'
            p[pf+'norm1_gamma'] = np.ones(d,   dtype=np.float32)
            p[pf+'norm1_beta']  = np.zeros(d,  dtype=np.float32)
            p[pf+'norm2_gamma'] = np.ones(d,   dtype=np.float32)
            p[pf+'norm2_beta']  = np.zeros(d,  dtype=np.float32)
            p[pf+'W_q'] = self._xavier(d, d)
            p[pf+'W_k'] = self._xavier(d, d)
            p[pf+'W_v'] = self._xavier(d, d)
            p[pf+'W_o'] = self._xavier(d, d)
            p[pf+'W1']  = self._xavier(d, dff)
            p[pf+'b1']  = np.zeros(dff, dtype=np.float32)
            p[pf+'W2']  = self._xavier(dff, d)
            p[pf+'b2']  = np.zeros(d,   dtype=np.float32)
        p['final_gamma'] = np.ones(d,  dtype=np.float32)
        p['final_beta']  = np.zeros(d, dtype=np.float32)
        p['head_W1'] = self._xavier(d, d // 2)
        p['head_b1'] = np.zeros(d // 2,              dtype=np.float32)
        p['head_W2'] = self._xavier(d // 2, self.num_aesthetics)
        p['head_b2'] = np.zeros(self.num_aesthetics, dtype=np.float32)
        return p

    def _layer_params(self, i):
        pf = f'l{i}_'
        p  = self.params
        return {k: p[pf+k] for k in
                ('norm1_gamma','norm1_beta','norm2_gamma','norm2_beta',
                 'W_q','W_k','W_v','W_o','W1','b1','W2','b2')}

    # ── Forward pass ──────────────────────────────────────────────────────

    def forward(self, patch_tokens: np.ndarray):
        """
        patch_tokens: float32 [B, 50, 768]
        Returns (logits [B, num_aesthetics], cache)
        """
        p = self.params

        # 1. Input projection
        x = patch_tokens @ p['proj_W'] + p['proj_b']   # [B, 50, d]

        # 2. Positional encoding
        x = x + p['pos_emb']

        # 3. Encoder layers
        layer_caches = []
        for i in range(self.num_layers):
            x, lc = encoder_layer_fwd(x, self._layer_params(i), self.num_heads)
            layer_caches.append(lc)

        # 4. CLS token + final norm
        cls      = x[:, 0, :]                                          # [B, d]
        cls_norm, final_ln_cache = layer_norm_fwd(
            cls, p['final_gamma'], p['final_beta']
        )

        # 5. MLP head
        h     = cls_norm @ p['head_W1'] + p['head_b1']                # [B, d//2]
        h_act = gelu(h)
        logits = h_act @ p['head_W2'] + p['head_b2']                  # [B, num_aes]

        cache = (patch_tokens, x, layer_caches, final_ln_cache, cls_norm, h, h_act)
        return logits, cache

    # ── Backward pass ─────────────────────────────────────────────────────

    def backward(self, d_logits: np.ndarray, cache) -> dict:
        """
        d_logits: [B, num_aesthetics]  gradient of loss w.r.t. logits
        Returns grads dict (same keys as self.params).
        """
        patch_tokens, x_final, layer_caches, final_ln_cache, cls_norm, h, h_act = cache
        p = self.params
        g = {}

        # 5. MLP head backward
        g['head_b2'] = d_logits.sum(axis=0)
        g['head_W2'] = h_act.T @ d_logits
        d_h_act      = d_logits @ p['head_W2'].T
        d_h          = d_h_act * gelu_grad(h)
        g['head_b1'] = d_h.sum(axis=0)
        g['head_W1'] = cls_norm.T @ d_h
        d_cls_norm   = d_h @ p['head_W1'].T

        # 4. Final layer norm backward
        d_cls, g['final_gamma'], g['final_beta'] = layer_norm_bwd(d_cls_norm, final_ln_cache)

        # Gradient enters x_final at position 0 only (CLS token)
        dx = np.zeros_like(x_final)
        dx[:, 0, :] = d_cls

        # 3. Encoder layers backward (reverse order)
        for i in range(self.num_layers - 1, -1, -1):
            pf = f'l{i}_'
            dx, layer_grads = encoder_layer_bwd(dx, layer_caches[i])
            for k, v in layer_grads.items():
                g[pf + k] = v

        # 2. Positional encoding: gradient passes straight through
        g['pos_emb'] = dx.sum(axis=0, keepdims=True)

        # 1. Input projection backward
        g['proj_b'] = dx.sum(axis=(0, 1))
        g['proj_W'] = patch_tokens.reshape(-1, self.CLIP_DIM).T @ dx.reshape(-1, self.d_model)

        return g

    # ── Adam optimiser step ───────────────────────────────────────────────

    def adam_step(self, grads: dict, lr=1e-4, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t += 1
        for key in self.params:
            grad = grads[key]
            self.m[key] = beta1 * self.m[key] + (1 - beta1) * grad
            self.v[key] = beta2 * self.v[key] + (1 - beta2) * grad ** 2
            m_hat = self.m[key] / (1 - beta1 ** self.t)
            v_hat = self.v[key] / (1 - beta2 ** self.t)
            self.params[key] -= lr * m_hat / (np.sqrt(v_hat) + eps)

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: str, aesthetic_names: list | None = None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = dict(self.params)
        if aesthetic_names:
            payload['__names__'] = np.array(aesthetic_names)
        np.savez(path, **payload)

    def load(self, path: str) -> list | None:
        data = np.load(path, allow_pickle=True)
        for k in self.params:
            if k in data:
                self.params[k] = data[k].astype(np.float32)
        # reset Adam state after loading
        self.m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.t = 0
        if '__names__' in data:
            return list(data['__names__'])
        return None


# ── Inference wrapper (frozen CLIP + NumPy transformer) ───────────────────────

def _score_label(score: float) -> str:
    if score >= 8.5:  return "Perfect match"
    if score >= 7.0:  return "Strong match"
    if score >= 5.5:  return "Good match"
    if score >= 4.0:  return "Moderate match"
    if score >= 2.5:  return "Weak match"
    return "Little to no match"


class AestheticScorerV2:
    """
    Frozen CLIP feature extractor (PyTorch) + NumPy AestheticTransformer.

    Scores images against aesthetics using the trained transformer.
    Drop-in complement to model.AestheticScorer.
    """

    def __init__(
        self,
        checkpoint_path: str | None = None,
        aesthetic_names: list[str] | None = None,
        device: str | None = None,
    ):
        if device is None:
            if torch.cuda.is_available():      device = "cuda"
            elif torch.backends.mps.is_available(): device = "mps"
            else:                              device = "cpu"
        self.device = device

        print(f"[AestheticScorerV2] Loading CLIP on {device} …")
        self.clip      = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip.eval()
        for p in self.clip.parameters():
            p.requires_grad = False

        self.transformer = AestheticTransformer()

        if checkpoint_path and os.path.exists(checkpoint_path):
            names = self.transformer.load(checkpoint_path)
            self.aesthetic_names: list[str] = names or aesthetic_names or []
            print(f"[AestheticScorerV2] Checkpoint loaded from {checkpoint_path}")
        else:
            self.aesthetic_names = aesthetic_names or []
            if checkpoint_path:
                print(f"[AestheticScorerV2] No checkpoint at {checkpoint_path}; random weights.")

    def extract_patch_tokens(self, images: list[Image.Image]) -> np.ndarray:
        """Return [B, 50, 768] float32 patch token array from CLIP vision backbone."""
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.clip.vision_model(pixel_values=inputs["pixel_values"])
        return out.last_hidden_state.cpu().numpy().astype(np.float32)

    def score_images(self, images: list[Image.Image]) -> np.ndarray:
        """Return aesthetic scores in [0, 10], shape [N, num_aesthetics]."""
        tokens = self.extract_patch_tokens(images)
        logits, _ = self.transformer.forward(tokens)
        return sigmoid(logits) * 10.0

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
