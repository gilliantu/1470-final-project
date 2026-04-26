"""
train_transformer.py — Train the AestheticTransformer.

Steps
-----
1. Download N scene images from fashion.json.
2. Generate soft labels using the prototype-based AestheticScorer
   (10 z-score aesthetic scores per image, values in [0, 10]).
3. Extract frozen CLIP patch tokens [N, 50, 768] (one-time pass).
4. Train AestheticTransformer with MSE loss (sigmoid × 10 output).
5. Save the best checkpoint to data/transformer.pt.

Feature cache (data/feature_cache.npz) is written after the first run.
Use --use-cache to skip downloading and CLIP extraction on re-runs.

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
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from aesthetics import AESTHETICS
from model import AestheticScorer, load_prototypes
from preprocess import batch_download, load_records, sample_random_scenes
from transformer_model import AestheticScorerV2, AestheticTransformer

PROTO_PATH = os.path.join(os.path.dirname(__file__), "data", "prototypes.npz")
CKPT_PATH  = os.path.join(os.path.dirname(__file__), "data", "transformer.pt")
CACHE_PATH = os.path.join(os.path.dirname(__file__), "data", "feature_cache.npz")


# ── Dataset ───────────────────────────────────────────────────────────────────

class AestheticDataset(Dataset):
    """(patch_tokens [50, 768], labels [10]) where labels are in [0, 10]."""

    def __init__(self, tokens: np.ndarray, labels: np.ndarray) -> None:
        self.tokens = torch.from_numpy(tokens).float()
        self.labels = torch.from_numpy(labels).float()

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, idx: int):
        return self.tokens[idx], self.labels[idx]


# ── Data helpers ──────────────────────────────────────────────────────────────

def generate_labels(scorer, prototypes, images, aesthetic_keys) -> np.ndarray:
    """Compute z-score prototype labels [N, 10] ∈ [0, 10]."""
    labels = []
    for i, img in enumerate(images):
        ranked    = scorer.rank_aesthetics(img, prototypes)
        score_map = {r["aesthetic"]: r["score"] for r in ranked}
        labels.append([score_map.get(k, 5.0) for k in aesthetic_keys])
        if (i + 1) % 50 == 0:
            print(f"  Labeled {i+1}/{len(images)}")
    return np.array(labels, dtype=np.float32)


def extract_patch_tokens(scorer_v2, images, batch_size: int = 16) -> np.ndarray:
    """Extract [N, 50, 768] CLIP patch tokens in mini-batches (numpy output)."""
    all_tokens = []
    for i in range(0, len(images), batch_size):
        batch  = images[i : i + batch_size]
        tokens = scorer_v2.extract_patch_tokens(batch).cpu().numpy()
        all_tokens.append(tokens)
        print(f"  Extracted {min(i + batch_size, len(images))}/{len(images)}", end="\r")
    print()
    return np.concatenate(all_tokens, axis=0)


# ── Training loop ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, device) -> float:
    model.train()
    total = 0.0
    for tokens, labels in loader:
        tokens, labels = tokens.to(device), labels.to(device)
        preds = torch.sigmoid(model(tokens)) * 10.0
        loss  = F.mse_loss(preds, labels)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    total = 0.0
    for tokens, labels in loader:
        tokens, labels = tokens.to(device), labels.to(device)
        preds = torch.sigmoid(model(tokens)) * 10.0
        total += F.mse_loss(preds, labels).item()
    return total / len(loader)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train the AestheticTransformer.")
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

    # ── Feature extraction + label generation ────────────────────────────
    if args.use_cache and os.path.exists(CACHE_PATH):
        print(f"Loading cache from {CACHE_PATH} …")
        cached      = np.load(CACHE_PATH)
        all_tokens  = cached["tokens"]
        all_labels  = cached["labels"]
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
        scorer     = AestheticScorer(device=device)
        all_labels = generate_labels(scorer, prototypes, images, aesthetic_keys)
        del scorer

        print("\nExtracting CLIP patch tokens …")
        sv2        = AestheticScorerV2(device=device)
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
    train_idx = idx[:split]
    val_idx   = idx[split:]

    train_ds = AestheticDataset(all_tokens[train_idx], all_labels[train_idx])
    val_ds   = AestheticDataset(all_tokens[val_idx],   all_labels[val_idx])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}\n")

    # ── Model ─────────────────────────────────────────────────────────────
    model    = AestheticTransformer(num_aesthetics=len(aesthetic_keys)).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"AestheticTransformer: {n_params:,} trainable parameters\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # ── Training ──────────────────────────────────────────────────────────
    best_val = float("inf")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    print(f"Training for {args.epochs} epochs …\n")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss   = evaluate(model, val_loader, device)
        scheduler.step()

        tag = ""
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict(), "aesthetic_names": aesthetic_keys},
                       args.output)
            tag = "  ← best"

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train={train_loss:.4f}  val={val_loss:.4f}{tag}")

    print(f"\nDone. Best val MSE: {best_val:.4f}")
    print(f"Checkpoint → {args.output}")


if __name__ == "__main__":
    main()
