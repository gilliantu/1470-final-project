"""
train_transformer.py — Train the NumPy AestheticTransformer.

Steps
-----
1. Download N scene images from fashion.json.
2. Generate soft labels (10 z-score aesthetic scores) using the existing
   prototype-based AestheticScorer.
3. Extract frozen CLIP patch tokens [N, 50, 768] (PyTorch, one-time pass).
4. Train the NumPy AestheticTransformer with manual backprop + Adam.
5. Save best checkpoint to data/transformer.npz.

Feature cache (data/feature_cache.npz) is written after the first run;
use --use-cache to skip downloading and CLIP extraction on re-runs.

Usage
-----
    python train_transformer.py
    python train_transformer.py --n-images 1000 --epochs 30
    python train_transformer.py --use-cache
"""

import argparse
import os
import random

import numpy as np

from aesthetics import AESTHETICS
from model import AestheticScorer, load_prototypes
from preprocess import batch_download, load_records, sample_random_scenes
from transformer_model import AestheticScorerV2, AestheticTransformer, sigmoid

PROTO_PATH = os.path.join(os.path.dirname(__file__), "data", "prototypes.npz")
CKPT_PATH  = os.path.join(os.path.dirname(__file__), "data", "transformer.npz")
CACHE_PATH = os.path.join(os.path.dirname(__file__), "data", "feature_cache.npz")


# ── Data preparation ──────────────────────────────────────────────────────────

def generate_labels(scorer, prototypes, images, aesthetic_keys):
    """
    For each image compute the prototype z-score for all aesthetics.
    Returns float32 [N, 10] array with values in [0, 10].
    """
    labels = []
    for i, img in enumerate(images):
        ranked    = scorer.rank_aesthetics(img, prototypes)
        score_map = {r["aesthetic"]: r["score"] for r in ranked}
        labels.append([score_map.get(k, 5.0) for k in aesthetic_keys])
        if (i + 1) % 50 == 0:
            print(f"  Labeled {i+1}/{len(images)}")
    return np.array(labels, dtype=np.float32)


def extract_patch_tokens(scorer_v2, images, batch_size=16):
    """
    Extract [N, 50, 768] CLIP patch tokens in mini-batches.
    CLIP weights are frozen; no gradients needed.
    """
    all_tokens = []
    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size]
        all_tokens.append(scorer_v2.extract_patch_tokens(batch))
        print(f"  Extracted {min(i+batch_size, len(images))}/{len(images)}", end="\r")
    print()
    return np.concatenate(all_tokens, axis=0)


# ── Loss ──────────────────────────────────────────────────────────────────────

def mse_sigmoid_loss(logits, targets):
    """
    MSE loss with sigmoid × 10 output.
    targets: [B, 10] in [0, 10].
    Returns (scalar loss, d_logits [B, 10]).
    """
    sig   = sigmoid(logits)
    preds = sig * 10.0
    diff  = preds - targets
    loss  = float(np.mean(diff ** 2))
    # dL/d_logits = 2/N * diff * 10 * sig * (1 - sig)
    d_logits = (2.0 / targets.size) * diff * 10.0 * sig * (1.0 - sig)
    return loss, d_logits.astype(np.float32)


# ── Training loop ─────────────────────────────────────────────────────────────

def run_epoch(model, tokens, labels, batch_size, lr, training=True):
    N = len(tokens)
    idx = np.random.permutation(N) if training else np.arange(N)
    total_loss = 0.0
    n_batches  = 0
    for start in range(0, N, batch_size):
        batch_idx = idx[start : start + batch_size]
        bt = tokens[batch_idx]   # [B, 50, 768]
        bl = labels[batch_idx]   # [B, 10]

        logits, cache = model.forward(bt)
        loss, d_logits = mse_sigmoid_loss(logits, bl)

        if training:
            grads = model.backward(d_logits, cache)
            model.adam_step(grads, lr=lr)

        total_loss += loss
        n_batches  += 1

    return total_loss / n_batches


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train the NumPy AestheticTransformer."
    )
    parser.add_argument("--n-images",   type=int,   default=500)
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--batch-size", type=int,   default=16)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--val-split",  type=float, default=0.1)
    parser.add_argument("--use-cache",  action="store_true",
                        help="Load data/feature_cache.npz instead of downloading.")
    parser.add_argument("--output",     type=str,   default=CKPT_PATH)
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    aesthetic_keys = list(AESTHETICS.keys())
    print(f"Aesthetics: {aesthetic_keys}\n")

    # Prototypes are needed for pseudo-label generation
    if not os.path.exists(PROTO_PATH):
        print(f"ERROR: run `python train.py` first to build {PROTO_PATH}")
        return
    prototypes = load_prototypes(PROTO_PATH)

    # ── Feature extraction + label generation ────────────────────────────
    if args.use_cache and os.path.exists(CACHE_PATH):
        print(f"Loading cache from {CACHE_PATH} …")
        cache_data  = np.load(CACHE_PATH)
        all_tokens  = cache_data["tokens"]
        all_labels  = cache_data["labels"]
        print(f"  {len(all_tokens)} cached samples.\n")
    else:
        print(f"Downloading {args.n_images} images …")
        records = load_records()
        urls    = sample_random_scenes(records, n=args.n_images, seed=args.seed)
        images  = batch_download(urls, verbose=True)
        if not images:
            print("No images downloaded — exiting.")
            return
        print(f"Downloaded {len(images)} images.\n")

        print("Generating pseudo-labels …")
        scorer     = AestheticScorer()
        all_labels = generate_labels(scorer, prototypes, images, aesthetic_keys)
        del scorer

        print("\nExtracting CLIP patch tokens …")
        sv2        = AestheticScorerV2()
        all_tokens = extract_patch_tokens(sv2, images)
        del sv2

        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        np.savez(CACHE_PATH, tokens=all_tokens, labels=all_labels)
        print(f"Cache saved → {CACHE_PATH}\n")

    print(f"Dataset: {len(all_tokens)} samples  "
          f"tokens={all_tokens.shape}  labels={all_labels.shape}\n")

    # ── Train / val split ─────────────────────────────────────────────────
    N   = len(all_tokens)
    idx = list(range(N))
    random.shuffle(idx)
    split     = max(1, int(N * (1 - args.val_split)))
    train_idx = np.array(idx[:split])
    val_idx   = np.array(idx[split:])

    train_tokens = all_tokens[train_idx]
    train_labels = all_labels[train_idx]
    val_tokens   = all_tokens[val_idx]
    val_labels   = all_labels[val_idx]
    print(f"Train: {len(train_tokens)} | Val: {len(val_tokens)}\n")

    # ── Model ─────────────────────────────────────────────────────────────
    model    = AestheticTransformer(num_aesthetics=len(aesthetic_keys), seed=args.seed)
    n_params = sum(v.size for v in model.params.values())
    print(f"AestheticTransformer: {n_params:,} parameters\n")

    # ── Training loop ─────────────────────────────────────────────────────
    best_val = float("inf")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(
            model, train_tokens, train_labels, args.batch_size, args.lr, training=True
        )
        val_loss = run_epoch(
            model, val_tokens,   val_labels,   args.batch_size, args.lr, training=False
        )

        tag = ""
        if val_loss < best_val:
            best_val = val_loss
            model.save(args.output, aesthetic_names=aesthetic_keys)
            tag = "  ← best"

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train={train_loss:.4f}  val={val_loss:.4f}{tag}")

    print(f"\nDone. Best val MSE: {best_val:.4f}")
    print(f"Checkpoint → {args.output}")


if __name__ == "__main__":
    main()
