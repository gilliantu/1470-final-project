"""
train_transformer.py — Train the custom AestheticTransformer.

How it works
------------
1. Download N scene images from fashion.json.
2. Use the existing prototype-based AestheticScorer to generate soft labels:
   for each image, compute z-score-normalized scores across all 10 aesthetics.
3. Extract 50×768 CLIP patch tokens (frozen) from each image.
4. Train AestheticTransformer with MSE loss: predict the same 10 scores.
5. Save the best checkpoint to data/transformer.pt.

A feature cache (data/feature_cache.npz) is written after the first run so
subsequent runs with --use-cache skip image downloading and CLIP extraction.

Usage
-----
    python train_transformer.py                    # 500 images, 20 epochs
    python train_transformer.py --n-images 1000 --epochs 30
    python train_transformer.py --use-cache        # skip download on re-run
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from aesthetics import AESTHETICS
from model import AestheticScorer, load_prototypes
from preprocess import batch_download, load_records, sample_random_scenes
from transformer_model import AestheticTransformer, AestheticScorerV2

PROTO_PATH = os.path.join(os.path.dirname(__file__), "data", "prototypes.npz")
CKPT_PATH  = os.path.join(os.path.dirname(__file__), "data", "transformer.pt")
CACHE_PATH = os.path.join(os.path.dirname(__file__), "data", "feature_cache.npz")


# ─── Dataset ──────────────────────────────────────────────────────────────────

class AestheticDataset(Dataset):
    """
    (patch_tokens [50, 768], labels [10])

    Labels are the prototype z-score aesthetic scores in [0, 10] produced by
    the existing AestheticScorer, used here as soft regression targets.
    """

    def __init__(self, patch_tokens: np.ndarray, labels: np.ndarray):
        self.patch_tokens = torch.from_numpy(patch_tokens).float()
        self.labels = torch.from_numpy(labels).float()

    def __len__(self) -> int:
        return len(self.patch_tokens)

    def __getitem__(self, idx: int):
        return self.patch_tokens[idx], self.labels[idx]


# ─── Data preparation ─────────────────────────────────────────────────────────

def generate_labels(
    scorer: AestheticScorer,
    prototypes: dict,
    images: list,
    aesthetic_keys: list[str],
) -> np.ndarray:
    """
    For each image, compute 10 z-score aesthetic scores via the prototype scorer.
    Returns float32 array of shape [N, 10] with values in [0, 10].
    """
    labels = []
    for i, img in enumerate(images):
        ranked = scorer.rank_aesthetics(img, prototypes)
        score_map = {r["aesthetic"]: r["score"] for r in ranked}
        row = [score_map.get(k, 5.0) for k in aesthetic_keys]
        labels.append(row)
        if (i + 1) % 50 == 0:
            print(f"  Labeled {i + 1}/{len(images)} images …")
    return np.array(labels, dtype=np.float32)


def extract_patch_tokens(
    scorer_v2: AestheticScorerV2,
    images: list,
    batch_size: int = 16,
) -> np.ndarray:
    """
    Extract [N, 50, 768] CLIP patch tokens in batches.
    No gradients; CLIP weights are frozen inside scorer_v2.
    """
    all_tokens = []
    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size]
        tokens = scorer_v2.extract_patch_tokens(batch)
        all_tokens.append(tokens.cpu().numpy())
        print(f"  Extracted {min(i + batch_size, len(images))}/{len(images)}", end="\r")
    print()
    return np.concatenate(all_tokens, axis=0)  # [N, 50, 768]


# ─── Training / evaluation ────────────────────────────────────────────────────

def train_epoch(
    model: AestheticTransformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> float:
    model.train()
    total = 0.0
    for patch_tokens, labels in loader:
        patch_tokens = patch_tokens.to(device)
        labels = labels.to(device)

        # Predict in [0, 10] via sigmoid × 10; targets already in [0, 10]
        preds = torch.sigmoid(model(patch_tokens)) * 10.0
        loss = nn.functional.mse_loss(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def evaluate(
    model: AestheticTransformer,
    loader: DataLoader,
    device: str,
) -> float:
    model.eval()
    total = 0.0
    for patch_tokens, labels in loader:
        patch_tokens = patch_tokens.to(device)
        labels = labels.to(device)
        preds = torch.sigmoid(model(patch_tokens)) * 10.0
        total += nn.functional.mse_loss(preds, labels).item()
    return total / len(loader)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train the custom AestheticTransformer on fashion.json images."
    )
    parser.add_argument(
        "--n-images", type=int, default=500,
        help="Images to download for training (default 500).",
    )
    parser.add_argument(
        "--epochs", type=int, default=20,
        help="Training epochs (default 20).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Mini-batch size (default 16).",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4,
        help="AdamW learning rate (default 1e-4).",
    )
    parser.add_argument(
        "--val-split", type=float, default=0.1,
        help="Fraction of data held out for validation (default 0.1).",
    )
    parser.add_argument(
        "--use-cache", action="store_true",
        help="Load cached patch tokens and labels from data/feature_cache.npz.",
    )
    parser.add_argument(
        "--output", type=str, default=CKPT_PATH,
        help=f"Checkpoint output path (default {CKPT_PATH}).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default 42).",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}")

    aesthetic_keys = list(AESTHETICS.keys())  # consistent order
    print(f"Aesthetics ({len(aesthetic_keys)}): {aesthetic_keys}")

    # ── Prototypes (for label generation) ────────────────────────────────
    if not os.path.exists(PROTO_PATH):
        print(f"\nERROR: {PROTO_PATH} not found. Run `python train.py` first.")
        return
    prototypes = load_prototypes(PROTO_PATH)

    # ── Feature extraction and label generation ───────────────────────────
    use_cache = args.use_cache and os.path.exists(CACHE_PATH)

    if use_cache:
        print(f"\nLoading cached features from {CACHE_PATH} …")
        cache = np.load(CACHE_PATH)
        all_tokens = cache["tokens"]   # [N, 50, 768]
        all_labels = cache["labels"]   # [N, 10]
        print(f"  Loaded {len(all_tokens)} cached samples.")
    else:
        print(f"\nDownloading {args.n_images} fashion.json scene images …")
        records = load_records()
        urls = sample_random_scenes(records, n=args.n_images, seed=args.seed)
        images = batch_download(urls, verbose=True)
        if not images:
            print("No images downloaded — exiting.")
            return
        print(f"Downloaded {len(images)} images.")

        print("\nGenerating pseudo-labels (prototype z-scores) …")
        scorer = AestheticScorer(device=device)
        all_labels = generate_labels(scorer, prototypes, images, aesthetic_keys)
        del scorer  # free memory before loading CLIP again inside V2

        print("\nExtracting CLIP patch tokens …")
        scorer_v2 = AestheticScorerV2(device=device)
        all_tokens = extract_patch_tokens(scorer_v2, images)
        del scorer_v2

        print(f"\nCaching features to {CACHE_PATH} …")
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        np.savez(CACHE_PATH, tokens=all_tokens, labels=all_labels)
        print("  Cache saved.")

    print(
        f"\nDataset: {len(all_tokens)} samples  "
        f"tokens={all_tokens.shape}  labels={all_labels.shape}"
    )

    # ── Train / val split ─────────────────────────────────────────────────
    N = len(all_tokens)
    idx = list(range(N))
    random.shuffle(idx)
    split = max(1, int(N * (1 - args.val_split)))
    train_idx, val_idx = idx[:split], idx[split:]

    train_ds = AestheticDataset(all_tokens[train_idx], all_labels[train_idx])
    val_ds   = AestheticDataset(all_tokens[val_idx],   all_labels[val_idx])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # ── Model ─────────────────────────────────────────────────────────────
    model = AestheticTransformer(num_aesthetics=len(aesthetic_keys)).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nAestheticTransformer: {n_params:,} trainable parameters")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # ── Training loop ─────────────────────────────────────────────────────
    best_val = float("inf")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    print(f"\nTraining for {args.epochs} epochs …\n")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss   = evaluate(model, val_loader, device)
        scheduler.step()

        tag = ""
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {"model": model.state_dict(), "aesthetic_names": aesthetic_keys},
                args.output,
            )
            tag = "  ← best"

        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"train={train_loss:.4f}  val={val_loss:.4f}{tag}"
        )

    print(f"\nDone. Best val MSE: {best_val:.4f}")
    print(f"Checkpoint saved to {args.output}")


if __name__ == "__main__":
    main()
