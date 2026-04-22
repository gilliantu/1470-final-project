"""
main.py — CLI for rating an outfit photo against a target aesthetic.

Usage examples
--------------
    # Rate an outfit against a specific aesthetic
    python main.py --image outfit.jpg --aesthetic streetwear

    # Let the model find the best matching aesthetic automatically
    python main.py --image outfit.jpg

    # List all available aesthetics
    python main.py --list

    # Rate an image from a URL
    python main.py --url http://example.com/outfit.jpg --aesthetic cottagecore

Workflow
--------
1. Run `python train.py` once to build prototype embeddings (data/prototypes.npz).
2. Then run `python main.py --image <path> --aesthetic <name>` to score.
"""

import argparse
import os
import sys

from PIL import Image

from model import AestheticScorer, load_prototypes
from preprocess import download_image
from aesthetics import AESTHETICS, AESTHETIC_NAMES

PROTO_PATH = os.path.join(os.path.dirname(__file__), "data", "prototypes.npz")


def print_result(result: dict, show_bar: bool = True) -> None:
    display = AESTHETIC_NAMES.get(result["aesthetic"], result["aesthetic"])
    score = result["score"]
    sim = result["similarity"]
    label = result["label"]

    print()
    print("╔══════════════════════════════════════════════╗")
    print(f"║  Aesthetic : {display:<32}║")
    print(f"║  Score     : {score:<5.1f} / 10  ({label:<16})║")
    print(f"║  Similarity: {sim:<32.4f}║")
    print("╚══════════════════════════════════════════════╝")

    if show_bar:
        filled = int(round(score))
        empty = 10 - filled
        bar = "▓" * filled + "░" * empty
        print(f"  [{bar}]  {score:.1f}/10")
    print()


def print_ranking(ranked: list[dict]) -> None:
    print()
    print("  ┌─ Aesthetic Ranking ──────────────────────────────────┐")
    for i, r in enumerate(ranked, 1):
        display = AESTHETIC_NAMES.get(r["aesthetic"], r["aesthetic"])
        bar_len = int(r["score"] / 10 * 25)
        bar = "█" * bar_len + "░" * (25 - bar_len)
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(i, f" {i}.")
        print(f"  │ {medal} {display:<18} {bar} {r['score']:>5.1f}/10")
    print("  └──────────────────────────────────────────────────────┘")
    best = ranked[0]
    best_display = AESTHETIC_NAMES.get(best["aesthetic"], best["aesthetic"])
    print(f"\n  Best match: {best_display} ({best['score']:.1f}/10 — {best['label']})\n")


def main():
    parser = argparse.ArgumentParser(
        description="Rate an outfit photo by aesthetic similarity (0–10).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --list
  python main.py --image outfit.jpg --aesthetic streetwear
  python main.py --image outfit.jpg          # rank all aesthetics
  python main.py --url <image_url> --aesthetic boho
        """,
    )
    parser.add_argument("--image", type=str, help="Path to a local outfit image.")
    parser.add_argument("--url", type=str, help="URL of an outfit image to download.")
    parser.add_argument(
        "--aesthetic", type=str, default=None,
        help="Target aesthetic key (see --list). Omit to rank all aesthetics.",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List all available aesthetics and exit.",
    )
    parser.add_argument(
        "--prototypes", type=str, default=PROTO_PATH,
        help=f"Path to prototypes .npz file (default {PROTO_PATH}).",
    )
    args = parser.parse_args()

    # ── List mode ─────────────────────────────────────────────────────
    if args.list:
        print("\nAvailable aesthetics:")
        for key, display in AESTHETIC_NAMES.items():
            prompts = AESTHETICS[key]
            print(f"  {key:<20} → {display}")
            print(f"    ({len(prompts)} reference prompts)")
        print()
        return

    # ── Validate input ─────────────────────────────────────────────────
    if not args.image and not args.url:
        parser.error("Provide --image or --url.")

    if args.aesthetic and args.aesthetic not in AESTHETICS:
        valid = ", ".join(AESTHETICS.keys())
        parser.error(f"Unknown aesthetic '{args.aesthetic}'. Valid: {valid}")

    # ── Load image ─────────────────────────────────────────────────────
    if args.image:
        if not os.path.exists(args.image):
            print(f"[ERROR] File not found: {args.image}")
            sys.exit(1)
        image = Image.open(args.image).convert("RGB")
        image_label = os.path.basename(args.image)
    else:
        print(f"Downloading image from URL …")
        image = download_image(args.url)
        if image is None:
            print("[ERROR] Could not download image. Check the URL and network.")
            sys.exit(1)
        image_label = args.url.split("/")[-1] or "downloaded_image"

    # ── Load prototypes ────────────────────────────────────────────────
    if not os.path.exists(args.prototypes):
        print(f"[ERROR] Prototype file not found: {args.prototypes}")
        print("Run `python train.py` first to build aesthetic prototypes.")
        sys.exit(1)

    prototypes = load_prototypes(args.prototypes)

    # ── Score ──────────────────────────────────────────────────────────
    scorer = AestheticScorer()

    print(f"\nImage: {image_label}")
    print(f"Size : {image.size[0]}×{image.size[1]} px")

    if args.aesthetic:
        # Single aesthetic scoring
        proto = prototypes[args.aesthetic]
        result = scorer.score_image(image, args.aesthetic, proto)
        print_result(result)
    else:
        # Rank all aesthetics
        print("\nRanking against all aesthetics …")
        ranked = scorer.rank_aesthetics(image, prototypes)
        print_ranking(ranked)


if __name__ == "__main__":
    main()
