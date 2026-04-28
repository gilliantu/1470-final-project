"""
test.py — Evaluate the aesthetic scorer on fashion.json scene images.

Usage
-----
    # Quick smoke-test: score 10 random fashion scenes vs all aesthetics
    python test.py

    # Score more images and write a CSV report
    python test.py --n 50 --output results/eval.csv

    # Test a single local image
    python test.py --image path/to/outfit.jpg

What it does
------------
1.  Loads prototype embeddings from data/prototypes.npz (run train.py first).
2.  Downloads N random scene images from fashion.json.
3.  For each image: scores it against every aesthetic, prints the top match,
    and (if --output) writes the full results to CSV.
4.  Reports aggregate statistics: how often each aesthetic "wins," and the
    mean similarity score per aesthetic (a sanity check that prototypes are
    calibrated and distinct from each other).
"""

import argparse
import csv
import os
import sys
from collections import Counter

from PIL import Image

from model import AestheticScorer, load_prototypes
from preprocess import (
    load_records,
    sample_random_scenes,
    download_image,
)
from aesthetics import AESTHETIC_NAMES

PROTO_PATH = os.path.join(os.path.dirname(__file__), "data", "prototypes.npz")


def evaluate(
    scorer: AestheticScorer,
    prototypes: dict,
    images: list[Image.Image],
    urls: list[str] | None = None,
) -> list[dict]:
    """Score every image against all aesthetics and return structured results."""
    all_results = []
    for i, img in enumerate(images):
        ranked = scorer.rank_aesthetics(img, prototypes)
        top = ranked[0]
        row = {
            "image_index": i,
            "url": urls[i] if urls else f"image_{i}",
            "top_aesthetic": top["aesthetic"],
            "top_score": top["score"],
            "top_similarity": top["similarity"],
        }
        for r in ranked:
            row[f"score_{r['aesthetic']}"] = r["score"]
        all_results.append(row)
    return all_results


def print_summary(results: list[dict], prototypes: dict) -> None:
    top_counts = Counter(r["top_aesthetic"] for r in results)
    print("\n" + "=" * 55)
    print("  AESTHETIC DISTRIBUTION (how often each wins)")
    print("=" * 55)
    for aes, cnt in top_counts.most_common():
        display = AESTHETIC_NAMES.get(aes, aes)
        bar = "█" * cnt
        print(f"  {display:<20} {bar} ({cnt})")

    print("\n" + "=" * 55)
    print("  MEAN SCORE PER AESTHETIC (across all test images)")
    print("=" * 55)
    for aes in prototypes:
        scores = [r[f"score_{aes}"] for r in results if f"score_{aes}" in r]
        mean = sum(scores) / len(scores) if scores else 0.0
        display = AESTHETIC_NAMES.get(aes, aes)
        print(f"  {display:<20} {mean:.2f}/10")


def save_csv(results: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved → {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the aesthetic scorer on fashion.json images."
    )
    parser.add_argument(
        "--n", type=int, default=10,
        help="Number of random scene images to download and evaluate (default 10).",
    )
    parser.add_argument(
        "--image", type=str, default=None,
        help="Path to a single local outfit image to score instead of downloading.",
    )
    parser.add_argument(
        "--prototypes", type=str, default=PROTO_PATH,
        help=f"Path to prototypes .npz file (default {PROTO_PATH}).",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Optional CSV path for detailed results.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for scene sampling (default 42).",
    )
    args = parser.parse_args()

    # Load prototypes
    if not os.path.exists(args.prototypes):
        print(f"[ERROR] Prototype file not found: {args.prototypes}")
        print("Run `python train.py` first to build the prototypes.")
        sys.exit(1)

    print(f"Loading prototypes from {args.prototypes} …")
    prototypes = load_prototypes(args.prototypes)
    print(f"Loaded {len(prototypes)} aesthetic prototypes: {list(prototypes.keys())}")

    scorer = AestheticScorer()

    # ── Single-image mode ──────────────────────────────────────────────
    if args.image:
        img = Image.open(args.image).convert("RGB")
        print(f"\nScoring: {args.image}")
        ranked = scorer.rank_aesthetics(img, prototypes)
        _print_ranked(ranked, image_label=os.path.basename(args.image))
        return

    # ── Dataset evaluation mode ────────────────────────────────────────
    print(f"\nLoading fashion.json dataset …")
    records = load_records()

    print(f"Sampling {args.n} random scene images …")
    urls = sample_random_scenes(records, n=args.n, seed=args.seed)

    print(f"Downloading images …")
    images = []
    valid_urls = []
    for url in urls:
        img = download_image(url)
        if img is not None:
            images.append(img)
            valid_urls.append(url)

    print(f"Downloaded {len(images)}/{len(urls)} images successfully.")

    if not images:
        print("[ERROR] No images could be downloaded. Check network access.")
        sys.exit(1)

    print("\nScoring images …")
    results = evaluate(scorer, prototypes, images, valid_urls)

    # Print individual results
    print("\n" + "=" * 55)
    print("  PER-IMAGE RESULTS")
    print("=" * 55)
    for r in results:
        top_display = AESTHETIC_NAMES.get(r["top_aesthetic"], r["top_aesthetic"])
        print(
            f"  [{r['image_index']:>3}] Top: {top_display:<20} "
            f"score={r['top_score']:.1f}/10  sim={r['top_similarity']:.4f}"
        )
        print(f"        {r['url']}")

    print_summary(results, prototypes)

    if args.output:
        save_csv(results, args.output)


def _print_ranked(ranked: list[dict], image_label: str = "outfit") -> None:
    print(f"\n{'─'*50}")
    print(f"  Aesthetic scores for: {image_label}")
    print(f"{'─'*50}")
    for r in ranked:
        display = AESTHETIC_NAMES.get(r["aesthetic"], r["aesthetic"])
        bar_len = int(r["score"] / 10 * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(
            f"  {display:<20} {bar}  {r['score']:>5.1f}/10  ({r['label']})"
        )
    print(f"{'─'*50}")
    best = ranked[0]
    best_display = AESTHETIC_NAMES.get(best["aesthetic"], best["aesthetic"])
    print(f"  Best match: {best_display} — {best['score']:.1f}/10")


if __name__ == "__main__":
    main()
