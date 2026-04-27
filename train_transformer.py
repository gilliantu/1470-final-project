"""
train_transformer.py — Train the AestheticTransformer.

Key design decisions
--------------------
Labels   : raw cosine similarities between CLIP image embedding and each prototype
           (values ~0.4–0.7, stable across images).  Z-score normalization is
           applied at inference time — same as the original prototype scorer.

Loss     : MSE  +  Pearson correlation
           MSE pushes absolute values toward the cosine similarities.
           Correlation maximises agreement in *ranking* across aesthetics,
           which is what the prototype scorer actually optimises for.

Features : both CLIP patch tokens [B, 50, 768] and the projected CLIP embedding
           [B, 512] are extracted and cached.  The projected embedding is already
           aligned with the text prototype space, giving the model the same
           information the original scorer uses.

Usage
-----
    python train_transformer.py
    python train_transformer.py --n-images 1500 --epochs 30
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
from transformer_model import (
    AestheticScorerV2, AestheticTransformer,
    AestheticScorerV3, AestheticTransformerV2, PretrainedPatchEmbedder,
    build_transform, build_train_transform,
)

PROTO_PATH       = os.path.join(os.path.dirname(__file__), "data", "prototypes.npz")
CKPT_PATH        = os.path.join(os.path.dirname(__file__), "data", "transformer.pt")
CKPT_PATH_V3     = os.path.join(os.path.dirname(__file__), "data", "transformer_v3.pt")
CACHE_PATH       = os.path.join(os.path.dirname(__file__), "data", "feature_cache.npz")
V3_CACHE_PATH    = os.path.join(os.path.dirname(__file__), "data", "v3_feature_cache.npz")


# ── Dataset ───────────────────────────────────────────────────────────────────

class AestheticDataset(Dataset):
    def __init__(
        self,
        tokens:   np.ndarray,   # [N, 50, 768]
        clip_embs: np.ndarray,  # [N, 512]
        labels:   np.ndarray,   # [N, 10]  raw cosine similarities
    ) -> None:
        self.tokens    = torch.from_numpy(tokens).float()
        self.clip_embs = torch.from_numpy(clip_embs).float()
        self.labels    = torch.from_numpy(labels).float()

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, idx: int):
        return self.tokens[idx], self.clip_embs[idx], self.labels[idx]


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


def extract_features(
    scorer_v2: AestheticScorerV2, images: list, batch_size: int = 16
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract patch tokens [N, 50, 768] and projected CLIP embeddings [N, 512].
    Both are needed by AestheticTransformer.forward.
    """
    all_tokens, all_embs = [], []
    for i in range(0, len(images), batch_size):
        batch       = images[i : i + batch_size]
        tokens, emb = scorer_v2.extract_features(batch)
        all_tokens.append(tokens.cpu().numpy())
        all_embs.append(emb.cpu().numpy())
        print(f"  Extracted {min(i+batch_size, len(images))}/{len(images)}", end="\r")
    print()
    return (
        np.concatenate(all_tokens, axis=0),
        np.concatenate(all_embs,   axis=0),
    )


# ── Training loop ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, device) -> float:
    model.train()
    total = 0.0
    for tokens, clip_embs, labels in loader:
        tokens    = tokens.to(device)
        clip_embs = clip_embs.to(device)
        labels    = labels.to(device)
        logits    = model(tokens, clip_embs)
        loss      = combined_loss(logits, labels)
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
    for tokens, clip_embs, labels in loader:
        tokens    = tokens.to(device)
        clip_embs = clip_embs.to(device)
        labels    = labels.to(device)
        logits    = model(tokens, clip_embs)
        total    += combined_loss(logits, labels).item()
    return total / len(loader)


# ── V3 dataset + loops (no CLIP at train/inference time) ─────────────────────

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
    parser = argparse.ArgumentParser(description="Train the AestheticTransformer.")
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
                        help="Load data/feature_cache.npz instead of downloading.")
    parser.add_argument("--output",        type=str,   default=CKPT_PATH)
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--model-version", type=str,   default="v2",
                        choices=["v2", "v3"],
                        help="v2 = CLIP features + AestheticTransformer (default); "
                             "v3 = CLIP-free PretrainedPatchEmbedder + AestheticTransformerV2.")
    parser.add_argument("--fine-tune-backbone", action="store_true",
                        help="(v3 only) Also fine-tune ResNet layer3+layer4, not just the "
                             "projection head. Slower but better with more data.")
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

    # ── V3 branch: CLIP-free multi-scale ResNet + AestheticTransformerV2 ──
    if args.model_version == "v3":
        out_path  = args.output if args.output != CKPT_PATH else CKPT_PATH_V3
        fine_tune = args.fine_tune_backbone

        # ── Images + labels ───────────────────────────────────────────────
        # For the frozen-backbone fast path we can reuse cached labels if
        # v3_feature_cache exists (labels are stored alongside features).
        v3_cache_ok = (not fine_tune) and args.use_cache and os.path.exists(V3_CACHE_PATH)

        if v3_cache_ok:
            print(f"Loading V3 feature cache from {V3_CACHE_PATH} …")
            cached_v3    = np.load(V3_CACHE_PATH)
            all_tokens   = cached_v3["tokens"]
            all_labels   = cached_v3["labels"]
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
                # Frozen backbone: extract features once and cache for reuse
                print("Extracting ResNet features (runs once, then cached) …")
                all_tokens = extract_v3_features(embedder, images, device)
                os.makedirs(os.path.dirname(V3_CACHE_PATH), exist_ok=True)
                np.savez(V3_CACHE_PATH, tokens=all_tokens, labels=all_labels)
                print(f"V3 feature cache saved → {V3_CACHE_PATH}\n")

        # ── Build model ───────────────────────────────────────────────────
        transformer = AestheticTransformerV2(num_aesthetics=len(aesthetic_keys)).to(device)
        n_params    = (sum(p.numel() for p in embedder.parameters() if p.requires_grad)
                       + sum(p.numel() for p in transformer.parameters() if p.requires_grad))
        backbone_mode = "fine-tuning layer3+layer4" if fine_tune else "frozen (cached)"
        print(f"PretrainedPatchEmbedder ({backbone_mode}) + AestheticTransformerV2: "
              f"{n_params:,} trainable parameters\n")

        # ── Train / val split ─────────────────────────────────────────────
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

        # ── Optimizer + schedule ──────────────────────────────────────────
        all_params   = list(embedder.parameters()) + list(transformer.parameters())
        optimizer    = torch.optim.AdamW(
            [p for p in all_params if p.requires_grad], lr=args.lr, weight_decay=1e-4,
        )
        total_steps  = args.epochs * len(train_loader)
        warmup_steps = int(total_steps * args.warmup)

        def lr_lambda_v3(step: int) -> float:
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(0.01, 0.5 * (1.0 + np.cos(np.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_v3)

        # ── Training loop ─────────────────────────────────────────────────
        best_val = float("inf")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        mode_str = "end-to-end" if fine_tune else "cached features (fast)"
        print(f"Training V3 ({mode_str}) for {args.epochs} epochs "
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
                    out_path,
                )
                tag = "  ← best"

            print(f"Epoch {epoch:3d}/{args.epochs}  "
                  f"train={train_loss:.4f}  val={val_loss:.4f}{tag}")

        print(f"\nDone. Best val loss: {best_val:.4f}")
        print(f"Checkpoint → {out_path}")
        return

    # ── V2 branch (original): CLIP features + AestheticTransformer ───────
    cache_ok = args.use_cache and os.path.exists(CACHE_PATH)
    if cache_ok:
        # Validate cache has the new clip_embs key
        cached = np.load(CACHE_PATH)
        if "clip_embs" not in cached:
            print("Cache is outdated (missing clip_embs) — re-extracting.")
            cache_ok = False

    if cache_ok:
        print(f"Loading cache from {CACHE_PATH} …")
        all_tokens    = cached["tokens"]
        all_clip_embs = cached["clip_embs"]
        all_labels    = cached["labels"]
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

        print("Generating cosine-similarity labels …")
        scorer     = AestheticScorer(device=device)
        all_labels = generate_labels(scorer, prototypes, images, aesthetic_keys)
        del scorer

        print("\nExtracting CLIP features (patch tokens + projected embedding) …")
        sv2                       = AestheticScorerV2(device=device)
        all_tokens, all_clip_embs = extract_features(sv2, images)
        del sv2

        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        np.savez(CACHE_PATH,
                 tokens=all_tokens, clip_embs=all_clip_embs, labels=all_labels)
        print(f"Cache saved → {CACHE_PATH}\n")

    print(f"Dataset: {len(all_tokens)} samples  "
          f"tokens={all_tokens.shape}  clip_embs={all_clip_embs.shape}  "
          f"labels={all_labels.shape}\n")
    print(f"Label range: {all_labels.min():.3f} – {all_labels.max():.3f}  "
          f"(cosine similarities)\n")

    # ── Train / val split ─────────────────────────────────────────────────
    N   = len(all_tokens)
    idx = list(range(N))
    random.shuffle(idx)
    split     = max(1, int(N * (1 - args.val_split)))
    train_idx = idx[:split]
    val_idx   = idx[split:]

    train_ds = AestheticDataset(all_tokens[train_idx], all_clip_embs[train_idx], all_labels[train_idx])
    val_ds   = AestheticDataset(all_tokens[val_idx],   all_clip_embs[val_idx],   all_labels[val_idx])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}\n")

    # ── Model ─────────────────────────────────────────────────────────────
    model    = AestheticTransformer(num_aesthetics=len(aesthetic_keys)).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"AestheticTransformer: {n_params:,} trainable parameters\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Cosine schedule with linear warmup
    total_steps  = args.epochs * len(train_loader)
    warmup_steps = int(total_steps * args.warmup)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.01, 0.5 * (1.0 + np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Training ──────────────────────────────────────────────────────────
    best_val = float("inf")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    print(f"Training for {args.epochs} epochs  "
          f"(warmup {args.warmup:.0%}, cosine decay) …\n")

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

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train={train_loss:.4f}  val={val_loss:.4f}{tag}")

    print(f"\nDone. Best val loss: {best_val:.4f}")
    print(f"Checkpoint → {args.output}")


if __name__ == "__main__":
    main()
