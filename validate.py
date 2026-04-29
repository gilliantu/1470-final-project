"""
validate.py — Quantitative evaluation: V3 Transformer vs. CLIP Prototype Baseline.

Scores a test set with both models and reports agreement, discrimination, and
per-aesthetic distribution metrics suitable for a results section.

Usage
-----
    python validate.py                      # personal_photos_for_testing/ (default)
    python validate.py --download 50        # download 50 images from fashion.json
    python validate.py --download 100 --seed 7
"""

import argparse
import os
import sys

import numpy as np
from PIL import Image

from aesthetics import AESTHETIC_NAMES
from model import AestheticScorer, load_prototypes
from transformer_model import AestheticScorerV3

PROTO_PATH   = os.path.join(os.path.dirname(__file__), "data", "prototypes.npz")
CKPT_PATH_V3 = os.path.join(os.path.dirname(__file__), "data", "transformer_v3.pt")
LOCAL_DIR    = os.path.join(os.path.dirname(__file__), "personal_photos_for_testing")
AESTHETIC_KEYS = list(AESTHETIC_NAMES.keys())
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".JPG", ".JPEG", ".PNG", ".WEBP"}


# ── Helpers ────────────────────────────────────────────────────────────────────

def spearman_r(a: np.ndarray, b: np.ndarray) -> float:
    ra = a.argsort().argsort().astype(float)
    rb = b.argsort().argsort().astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def scores_to_row(ranked: list[dict]) -> list[float]:
    lookup = {r["aesthetic"]: r["score"] for r in ranked}
    return [lookup.get(k, 5.0) for k in AESTHETIC_KEYS]


def score_all(clip_scorer, v3_scorer, prototypes, images, labels):
    clip_mat, v3_mat = [], []
    n = len(images)
    for i, img in enumerate(images):
        print(f"  [{i+1:>3}/{n}] {labels[i]:<35}", end="\r")
        clip_mat.append(scores_to_row(clip_scorer.rank_aesthetics(img, prototypes)))
        v3_mat.append(scores_to_row(v3_scorer.rank_aesthetics(img)))
    print()
    return np.array(clip_mat, dtype=np.float32), np.array(v3_mat, dtype=np.float32)


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(clip: np.ndarray, v3: np.ndarray) -> dict:
    N = len(clip)
    corrs     = np.array([spearman_r(clip[i], v3[i]) for i in range(N)])
    clip_top1 = clip.argmax(axis=1)
    v3_top1   = v3.argmax(axis=1)
    top1_agree = int(np.sum(clip_top1 == v3_top1))

    top3_agree = 0
    for i in range(N):
        clip_top3 = set(np.argsort(clip[i])[-3:])
        v3_top3   = set(np.argsort(v3[i])[-3:])
        if v3_top1[i] in clip_top3 or clip_top1[i] in v3_top3:
            top3_agree += 1

    def top2_margin(mat):
        sorted_desc = np.sort(mat, axis=1)[:, ::-1]
        return sorted_desc[:, 0] - sorted_desc[:, 1]

    return {
        "N":               N,
        "corrs":           corrs,
        "top1_agree":      top1_agree,
        "top3_agree":      top3_agree,
        "clip_margin":     top2_margin(clip),
        "v3_margin":       top2_margin(v3),
        "clip_top1_freq":  [int(np.sum(clip_top1 == i)) for i in range(len(AESTHETIC_KEYS))],
        "v3_top1_freq":    [int(np.sum(v3_top1 == i)) for i in range(len(AESTHETIC_KEYS))],
        "clip_mean":       clip.mean(axis=0),
        "v3_mean":         v3.mean(axis=0),
        "clip_top1_idx":   clip_top1,
        "v3_top1_idx":     v3_top1,
    }


# ── Report ────────────────────────────────────────────────────────────────────

def print_report(m: dict, labels: list[str]) -> None:
    N  = m["N"]
    W  = 66
    hr = "  " + "─" * (W - 2)

    print()
    print("═" * W)
    print("  AESTHETIC SCORER — QUANTITATIVE VALIDATION")
    print("  V3 Transformer  vs.  CLIP Prototype Baseline")
    print(f"  N = {N} images")
    print("═" * W)

    # ── Agreement ─────────────────────────────────────────────────────────────
    corrs = m["corrs"]
    print()
    print("  1. MODEL AGREEMENT")
    print(hr)
    print(f"  {'Metric':<40} {'Value':>10}")
    print(hr)
    print(f"  {'Spearman rank correlation (mean ± std)':<40} {corrs.mean():.3f} ± {corrs.std():.3f}")
    print(f"  {'Spearman correlation (median)':<40} {np.median(corrs):>10.3f}")
    print(f"  {'Top-1 aesthetic agreement':<40} {m['top1_agree']/N*100:>9.1f}%  ({m['top1_agree']}/{N})")
    print(f"  {'Top-3 aesthetic agreement':<40} {m['top3_agree']/N*100:>9.1f}%  ({m['top3_agree']}/{N})")

    # ── Confidence margin ─────────────────────────────────────────────────────
    cm, vm = m["clip_margin"], m["v3_margin"]
    print()
    print("  2. CONFIDENCE MARGIN  (top-1 score minus top-2 score per image)")
    print(hr)
    print(f"  {'Metric':<28} {'CLIP Baseline':>16} {'V3 Transformer':>16}")
    print(hr)
    print(f"  {'Mean margin':<28} {cm.mean():>16.3f} {vm.mean():>16.3f}")
    print(f"  {'Std of margin':<28} {cm.std():>16.3f} {vm.std():>16.3f}")
    print(f"  {'Min margin':<28} {cm.min():>16.3f} {vm.min():>16.3f}")
    print(f"  {'Max margin':<28} {cm.max():>16.3f} {vm.max():>16.3f}")

    # ── Per-aesthetic frequency ───────────────────────────────────────────────
    print()
    print("  3. TOP-1 FREQUENCY PER AESTHETIC")
    print(hr)
    print(f"  {'Aesthetic':<20} {'CLIP Baseline':>16} {'V3 Transformer':>16}")
    print(hr)
    for i, key in enumerate(AESTHETIC_KEYS):
        cf = m["clip_top1_freq"][i]
        vf = m["v3_top1_freq"][i]
        bar_c = "▓" * cf
        bar_v = "▓" * vf
        print(f"  {AESTHETIC_NAMES[key]:<20} {cf:>5}  {bar_c:<10} {vf:>5}  {bar_v:<10}")

    # ── Mean score per aesthetic ──────────────────────────────────────────────
    print()
    print("  4. MEAN SCORE PER AESTHETIC  (0–10 scale, across all test images)")
    print(hr)
    print(f"  {'Aesthetic':<20} {'CLIP Baseline':>16} {'V3 Transformer':>16} {'  Δ':>6}")
    print(hr)
    for i, key in enumerate(AESTHETIC_KEYS):
        cm = m["clip_mean"][i]
        vm = m["v3_mean"][i]
        print(f"  {AESTHETIC_NAMES[key]:<20} {cm:>16.2f} {vm:>16.2f} {abs(cm-vm):>6.2f}")

    # ── Per-image predictions ─────────────────────────────────────────────────
    print()
    print("  5. PER-IMAGE TOP AESTHETIC PREDICTIONS")
    print(hr)
    print(f"  {'Image':<35} {'CLIP Top-1':<20} {'V3 Top-1':<20} {'Match':>5}")
    print(hr)
    for i, label in enumerate(labels):
        ci = m["clip_top1_idx"][i]
        vi = m["v3_top1_idx"][i]
        c_aes = AESTHETIC_NAMES[AESTHETIC_KEYS[ci]]
        v_aes = AESTHETIC_NAMES[AESTHETIC_KEYS[vi]]
        match = "✓" if ci == vi else "✗"
        short = label[:33] + ".." if len(label) > 35 else label
        print(f"  {short:<35} {c_aes:<20} {v_aes:<20} {match:>5}")

    print()
    print("═" * W)
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Quantitative validation: V3 Transformer vs. CLIP Baseline."
    )
    parser.add_argument(
        "--download", type=int, default=None, metavar="N",
        help="Download N images from fashion.json instead of using local photos.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    for path, name in [(PROTO_PATH, "data/prototypes.npz"), (CKPT_PATH_V3, "data/transformer_v3.pt")]:
        if not os.path.exists(path):
            print(f"[ERROR] {name} not found. Run train.py / train_transformer.py first.")
            sys.exit(1)

    print("\nLoading CLIP baseline …")
    prototypes  = load_prototypes(PROTO_PATH)
    clip_scorer = AestheticScorer()

    print("Loading V3 transformer …")
    v3_scorer = AestheticScorerV3(
        checkpoint_path=CKPT_PATH_V3,
        aesthetic_names=AESTHETIC_KEYS,
    )

    if args.download:
        from preprocess import load_records, sample_random_scenes, batch_download
        fashion_json = os.path.join(os.path.dirname(__file__), "fashion.json", "fashion.json")
        if not os.path.exists(fashion_json):
            print("[ERROR] fashion.json dataset not found. Omit --download to use local photos.")
            sys.exit(1)
        print(f"\nDownloading {args.download} test images (seed={args.seed}) …")
        records = load_records()
        urls    = sample_random_scenes(records, n=args.download, seed=args.seed)
        images  = batch_download(urls, verbose=True)
        if not images:
            print("[ERROR] No images downloaded.")
            sys.exit(1)
        labels = [f"scene_{i+1:03d}" for i in range(len(images))]
        print(f"Downloaded {len(images)}/{args.download} images.\n")
    else:
        paths = sorted(
            p for p in (
                os.path.join(LOCAL_DIR, f) for f in os.listdir(LOCAL_DIR)
            )
            if os.path.splitext(p)[1] in IMG_EXTS
        )
        if not paths:
            print(f"[ERROR] No images found in {LOCAL_DIR}")
            sys.exit(1)
        images = [Image.open(p).convert("RGB") for p in paths]
        labels = [os.path.basename(p) for p in paths]
        print(f"\nLoaded {len(images)} images from personal_photos_for_testing/\n")

    print("Scoring with both models …")
    clip_scores, v3_scores = score_all(clip_scorer, v3_scorer, prototypes, images, labels)

    metrics = compute_metrics(clip_scores, v3_scores)
    print_report(metrics, labels)


if __name__ == "__main__":
    main()
