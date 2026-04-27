"""
train_transformer.py — Train the V3 AestheticTransformer (CLIP-free).

Key design decisions
--------------------
Labels   : raw cosine similarities between CLIP image embedding and each prototype
           (values ~0.4–0.7, stable across images).  Z-score normalization is
           applied at inference time — same as the original prototype scorer.

Loss     : MSE  +  Pearson correlation
           MSE pushes absolute values toward the cosine similarities.
           Correlation maximises agreement in *ranking* across aesthetics,
           which is what the prototype scorer actually optimises for.

Features : ResNet-50 multi-scale patch tokens [B, 50, 768] extracted once
           (frozen backbone) and cached, or extracted per-step (fine-tune mode).

Usage
-----
    python train_transformer.py
    python train_transformer.py --n-images 1500 --epochs 30
    python train_transformer.py --use-cache
    python train_transformer.py --fine-tune-backbone
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from aesthetics import AESTHETICS
from model import AestheticScorer, load_prototypes
from preprocess import batch_download, load_records, sample_random_scenes
from transformer_model import (
    AestheticTransformerV2, PretrainedPatchEmbedder,
    build_transform, build_train_transform,
)

PROTO_PATH    = os.path.join(os.path.dirname(__file__), "data", "prototypes.npz")
CKPT_PATH_V3  = os.path.join(os.path.dirname(__file__), "data", "transformer_v3.pt")
V3_CACHE_PATH = os.path.join(os.path.dirname(__file__), "data", "v3_feature_cache.npz")


# ── Loss ──────────────────────────────────────────────────────────────────────

def pearson_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    1 - mean Pearson correlation across the batch.
    Optimises the *ranking* of aesthetic scores per image, independent of
    absolute scale — directly mirrors what the prototype scorer does.
    """
    pred_c   = pred   - pred.mean(dim=-1,   keepdim=True)
    target_c = target - target.mean(dim=-1, keepdim=True)
    corr = (pred_c * target_c).sum(dim=-1) / (
        pred_c.norm(dim=-1) * target_c.norm(dim=-1) + 1e-8
    )
    return 1.0 - corr.mean()


def combined_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """MSE on absolute values + Pearson on relative ranking."""
    return F.mse_loss(logits, targets) + 0.5 * pearson_loss(logits, targets)


# ── Data helpers ──────────────────────────────────────────────────────────────

def generate_labels(
    scorer, prototypes: dict, images: list, aesthetic_keys: list[str]
) -> np.ndarray:
    """
    Raw cosine similarities [N, 10] ∈ [0.4, 0.7].
    More stable than z-scores as training targets; z-score normalisation
    is applied at inference time instead.
    """
    labels = []
    for i, img in enumerate(images):
        emb = scorer.encode_image(img)   # L2-normalised [512]
        row = [float(np.dot(emb, prototypes[k])) for k in aesthetic_keys]
        labels.append(row)
        if (i + 1) % 50 == 0:
            print(f"  Labeled {i+1}/{len(images)}")
    return np.array(labels, dtype=np.float32)


# ── Dataset + training loops ──────────────────────────────────────────────────

class AestheticDatasetV3(Dataset):
    """
    Holds PIL images + cosine-similarity labels.
    PatchEmbedder runs inside the training loop so both embedder and
    transformer are trained end-to-end on the same backward pass.
    """
    def __init__(self, images: list, labels: np.ndarray, transform) -> None:
        self.images    = images
        self.labels    = torch.from_numpy(labels).float()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img = self.images[idx].convert("RGB")
        return self.transform(img), self.labels[idx]


def train_epoch_v3(embedder, transformer, loader, optimizer, device) -> float:
    embedder.train()
    transformer.train()
    total = 0.0
    for pixel_values, labels in loader:
        pixel_values = pixel_values.to(device)
        labels       = labels.to(device)
        tokens = embedder(pixel_values)     # [B, 50, patch_dim]
        logits = transformer(tokens)        # [B, num_aesthetics]
        loss   = combined_loss(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(embedder.parameters()) + list(transformer.parameters()), 1.0
        )
        optimizer.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def evaluate_v3(embedder, transformer, loader, device) -> float:
    embedder.eval()
    transformer.eval()
    total = 0.0
    for pixel_values, labels in loader:
        pixel_values = pixel_values.to(device)
        labels       = labels.to(device)
        logits = transformer(embedder(pixel_values))
        total += combined_loss(logits, labels).item()
    return total / len(loader)


class AestheticDatasetV3Cache(Dataset):
    """Pre-extracted patch tokens + labels — no backbone forward pass per step."""
    def __init__(self, tokens: np.ndarray, labels: np.ndarray) -> None:
        self.tokens = torch.from_numpy(tokens).float()
        self.labels = torch.from_numpy(labels).float()

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, idx: int):
        return self.tokens[idx], self.labels[idx]


def extract_v3_features(embedder, images: list, device: str, batch_size: int = 32) -> np.ndarray:
    """Run all images through frozen PretrainedPatchEmbedder → [N, 50, patch_dim]."""
    transform = build_transform()
    embedder.eval()
    all_tokens = []
    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size]
        pixel_values = torch.stack(
            [transform(img.convert("RGB")) for img in batch]
        ).to(device)
        with torch.no_grad():
            tokens = embedder(pixel_values)
        all_tokens.append(tokens.cpu().numpy())
        print(f"  Extracted {min(i + batch_size, len(images))}/{len(images)}", end="\r")
    print()
    return np.concatenate(all_tokens, axis=0)


def train_epoch_v3_cached(transformer, loader, optimizer, device) -> float:
    transformer.train()
    total = 0.0
    for tokens, labels in loader:
        tokens = tokens.to(device)
        labels = labels.to(device)
        logits = transformer(tokens)
        loss   = combined_loss(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(transformer.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def evaluate_v3_cached(transformer, loader, device) -> float:
    transformer.eval()
    total = 0.0
    for tokens, labels in loader:
        tokens = tokens.to(device)
        labels = labels.to(device)
        total += combined_loss(transformer(tokens), labels).item()
    return total / len(loader)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train the V3 AestheticTransformer (CLIP-free).")
    parser.add_argument("--n-images",      type=int,   default=1500,
                        help="Images to download for training (default 1500).")
    parser.add_argument("--epochs",        type=int,   default=30,
                        help="Training epochs (default 30).")
    parser.add_argument("--batch-size",    type=int,   default=32)
    parser.add_argument("--lr",            type=float, default=2e-4)
    parser.add_argument("--val-split",     type=float, default=0.1)
    parser.add_argument("--warmup",        type=float, default=0.1,
                        help="Fraction of steps used for linear LR warmup (default 0.1).")
    parser.add_argument("--use-cache",     action="store_true",
                        help="Load data/v3_feature_cache.npz instead of downloading.")
    parser.add_argument("--output",        type=str,   default=CKPT_PATH_V3)
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--fine-tune-backbone", action="store_true",
                        help="Also fine-tune ResNet layer3+layer4 (slower, better with more data).")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():           device = "cuda"
    elif torch.backends.mps.is_available(): device = "mps"
    else:                                   device = "cpu"
    print(f"Device: {device}\n")

    aesthetic_keys = list(AESTHETICS.keys())

    if not os.path.exists(PROTO_PATH):
        print(f"ERROR: run `python train.py` first to build {PROTO_PATH}")
        return
    prototypes = load_prototypes(PROTO_PATH)

    fine_tune   = args.fine_tune_backbone
    cache_ok    = (not fine_tune) and args.use_cache and os.path.exists(V3_CACHE_PATH)

    # ── Images + labels ───────────────────────────────────────────────────
    if cache_ok:
        print(f"Loading feature cache from {V3_CACHE_PATH} …")
        cached_v3  = np.load(V3_CACHE_PATH)
        all_tokens = cached_v3["tokens"]
        all_labels = cached_v3["labels"]
        print(f"  {len(all_tokens)} cached samples.\n")
        embedder = PretrainedPatchEmbedder(fine_tune=False).to(device)
    else:
        print(f"Downloading {args.n_images} images …")
        records = load_records()
        urls    = sample_random_scenes(records, n=args.n_images, seed=args.seed)
        images  = batch_download(urls, verbose=True)
        if not images:
            print("No images downloaded — exiting.")
            return
        print(f"Downloaded {len(images)} images.\n")

        print("Generating cosine-similarity labels (CLIP used for labels only) …")
        scorer     = AestheticScorer(device=device)
        all_labels = generate_labels(scorer, prototypes, images, aesthetic_keys)
        del scorer
        print(f"Labels: {all_labels.shape}  range {all_labels.min():.3f}–{all_labels.max():.3f}\n")

        embedder = PretrainedPatchEmbedder(fine_tune=fine_tune).to(device)

        if not fine_tune:
            print("Extracting ResNet features (runs once, then cached) …")
            all_tokens = extract_v3_features(embedder, images, device)
            os.makedirs(os.path.dirname(V3_CACHE_PATH), exist_ok=True)
            np.savez(V3_CACHE_PATH, tokens=all_tokens, labels=all_labels)
            print(f"Feature cache saved → {V3_CACHE_PATH}\n")

    # ── Build model ───────────────────────────────────────────────────────
    transformer = AestheticTransformerV2(num_aesthetics=len(aesthetic_keys)).to(device)
    n_params    = (sum(p.numel() for p in embedder.parameters() if p.requires_grad)
                   + sum(p.numel() for p in transformer.parameters() if p.requires_grad))
    backbone_mode = "fine-tuning layer3+layer4" if fine_tune else "frozen (cached)"
    print(f"PretrainedPatchEmbedder ({backbone_mode}) + AestheticTransformerV2: "
          f"{n_params:,} trainable parameters\n")

    # ── Train / val split ─────────────────────────────────────────────────
    N   = len(all_tokens) if not fine_tune else len(images)
    idx = list(range(N))
    random.shuffle(idx)
    split     = max(1, int(N * (1 - args.val_split)))
    train_idx = idx[:split]
    val_idx   = idx[split:]

    if not fine_tune:
        train_ds = AestheticDatasetV3Cache(all_tokens[train_idx], all_labels[train_idx])
        val_ds   = AestheticDatasetV3Cache(all_tokens[val_idx],   all_labels[val_idx])
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)
    else:
        train_ds = AestheticDatasetV3(
            [images[i] for i in train_idx], all_labels[train_idx], build_train_transform(),
        )
        val_ds = AestheticDatasetV3(
            [images[i] for i in val_idx],   all_labels[val_idx],   build_transform(),
        )
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
        val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}\n")

    # ── Optimizer + schedule ──────────────────────────────────────────────
    # Backbone (if fine-tuning) gets 10× smaller lr to avoid destructive
    # updates to pretrained ResNet weights.
    backbone_params = [p for p in embedder.parameters() if p.requires_grad]
    head_params     = list(transformer.parameters())
    param_groups    = [
        {"params": backbone_params, "lr": args.lr * 0.1},
        {"params": head_params,     "lr": args.lr},
    ]
    optimizer    = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    total_steps  = args.epochs * len(train_loader)
    warmup_steps = int(total_steps * args.warmup)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.01, 0.5 * (1.0 + np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Training loop ─────────────────────────────────────────────────────
    best_val = float("inf")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    mode_str = "end-to-end" if fine_tune else "cached features (fast)"
    print(f"Training ({mode_str}) for {args.epochs} epochs "
          f"(warmup {args.warmup:.0%}, cosine decay) …\n")

    for epoch in range(1, args.epochs + 1):
        if not fine_tune:
            train_loss = train_epoch_v3_cached(transformer, train_loader, optimizer, device)
            val_loss   = evaluate_v3_cached(transformer, val_loader, device)
        else:
            train_loss = train_epoch_v3(embedder, transformer, train_loader, optimizer, device)
            val_loss   = evaluate_v3(embedder, transformer, val_loader, device)
        scheduler.step()

        tag = ""
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "embedder":        embedder.state_dict(),
                    "transformer":     transformer.state_dict(),
                    "aesthetic_names": aesthetic_keys,
                },
                args.output,
            )
            tag = "  ← best"

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train={train_loss:.4f}  val={val_loss:.4f}{tag}")

    print(f"\nDone. Best val loss: {best_val:.4f}")
    print(f"Checkpoint → {args.output}")


if __name__ == "__main__":
    main()
