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
1. Run `python train.py` then `python train_transformer.py` to build the model.
2. Then run `python main.py --image <path> --aesthetic <name>` to score.
"""

import argparse
import os
import sys

from PIL import Image

from transformer_model import AestheticScorerV3
from preprocess import download_image
from aesthetics import AESTHETICS, AESTHETIC_NAMES

CKPT_PATH_V3 = os.path.join(os.path.dirname(__file__), "data", "transformer_v3.pt")


def print_result(result: dict, show_bar: bool = True) -> None:
    display = AESTHETIC_NAMES.get(result["aesthetic"], result["aesthetic"])
    score = result["score"]
    label = result["label"]

    print()
    print("╔══════════════════════════════════════════════╗")
    print(f"║  Aesthetic : {display:<32}║")
    print(f"║  Score     : {score:<5.1f} / 10  ({label:<16})║")
    print("╚══════════════════════════════════════════════╝")

    if show_bar:
        filled = int(round(score))
        empty = 10 - filled
        bar = "▓" * filled + "░" * empty
        print(f"  [{bar}]  {score:.1f}/10")
    print()


def print_ranking(ranked: list[dict]) -> None:
    print()
    print("  ┌─ Aesthetic Ranking ──────────────────────────────────────┐")
    for i, r in enumerate(ranked, 1):
        display = AESTHETIC_NAMES.get(r["aesthetic"], r["aesthetic"])
        bar_len = int(r["score"] / 10 * 25)
        bar = "█" * bar_len + "░" * (25 - bar_len)
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(i, f" {i}.")
        print(f"  │ {medal} {display:<18} {bar} {r['score']:>5.1f}/10")
    print("  └──────────────────────────────────────────────────────────┘")
    best = ranked[0]
    best_display = AESTHETIC_NAMES.get(best["aesthetic"], best["aesthetic"])
    print(f"\n  Best match: {best_display} ({best['score']:.1f}/10 — {best['label']})\n")


def print_recommendation(base_path: str, item_paths: list[str], scorer) -> None:
    base_image = Image.open(base_path).convert("RGB")

    item_images = []
    for path in item_paths:
        if not os.path.exists(path):
            print(f"[ERROR] Item image not found: {path}")
            sys.exit(1)
        item_images.append((os.path.basename(path), Image.open(path).convert("RGB")))

    print(f"\nBase outfit : {os.path.basename(base_path)}")
    print(f"Items       : {len(item_images)}")

    print("\nAnalyzing base outfit …")
    base_ranked = scorer.rank_aesthetics(base_image)
    base_top = base_ranked[0]
    base_top_key = base_top["aesthetic"]
    base_top_display = AESTHETIC_NAMES.get(base_top_key, base_top_key)
    print(f"  → {base_top_display}  ({base_top['score']:.1f}/10  {base_top['label']})")

    print("\nAnalyzing items …")
    item_results = []
    for name, img in item_images:
        ranked = scorer.rank_aesthetics(img)
        item_top = ranked[0]
        base_score = next((r["score"] for r in ranked if r["aesthetic"] == base_top_key), 0.0)
        matches = item_top["aesthetic"] == base_top_key
        item_results.append({
            "name": name,
            "top_display": AESTHETIC_NAMES.get(item_top["aesthetic"], item_top["aesthetic"]),
            "base_score": base_score,
            "matches": matches,
        })
        tag = "✓ MATCH" if matches else "✗"
        top_disp = AESTHETIC_NAMES.get(item_top["aesthetic"], item_top["aesthetic"])
        print(f"  {name:<35} → {top_disp:<20} {tag}")

    exact = [r for r in item_results if r["matches"]]
    if exact:
        rec = max(exact, key=lambda r: r["base_score"])
        match_kind = "exact aesthetic match"
    else:
        rec = max(item_results, key=lambda r: r["base_score"])
        match_kind = "closest aesthetic (no exact match)"

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print(f"║  RECOMMENDED ITEM  ({match_kind})")
    w = 60
    for line in [rec["name"], f"Aesthetic: {rec['top_display']}",
                 f"Score for {base_top_display}: {rec['base_score']:.1f}/10"]:
        print(f"║  {line:<{w-4}}║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()


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
  python main.py --recommend --base outfit.jpg --items item1.jpg item2.jpg item3.jpg
        """,
    )
    parser.add_argument("--image", type=str, help="Path to a local outfit image.")
    parser.add_argument("--url",   type=str, help="URL of an outfit image to download.")
    parser.add_argument(
        "--aesthetic", type=str, default=None,
        help="Target aesthetic key (see --list). Omit to rank all aesthetics.",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List all available aesthetics and exit.",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to a custom checkpoint (default: data/transformer_v3.pt).",
    )
    parser.add_argument(
        "--recommend", action="store_true",
        help="Recommend which item best matches the base outfit's aesthetic.",
    )
    parser.add_argument(
        "--base", type=str,
        help="(--recommend mode) Path to the base outfit image.",
    )
    parser.add_argument(
        "--items", type=str, nargs="+",
        help="(--recommend mode) Paths to individual clothing item images.",
    )
    args = parser.parse_args()

    # ── List mode ─────────────────────────────────────────────────────
    if args.list:
        print("\nAvailable aesthetics:")
        for key, display in AESTHETIC_NAMES.items():
            print(f"  {key:<20} → {display}")
        print()
        return

    # ── Recommend mode ─────────────────────────────────────────────────
    if args.recommend:
        if not args.base:
            parser.error("--recommend requires --base <outfit image>.")
        if not args.items or len(args.items) < 1:
            parser.error("--recommend requires at least one --items image.")
        if not os.path.exists(args.base):
            print(f"[ERROR] Base image not found: {args.base}")
            sys.exit(1)
        ckpt = args.checkpoint or CKPT_PATH_V3
        if not os.path.exists(ckpt):
            print(f"[ERROR] Checkpoint not found: {ckpt}")
            print("Run: python train_transformer.py")
            sys.exit(1)
        scorer = AestheticScorerV3(
            checkpoint_path=ckpt,
            aesthetic_names=list(AESTHETIC_NAMES.keys()),
        )
        print_recommendation(args.base, args.items, scorer)
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
        print("Downloading image from URL …")
        image = download_image(args.url)
        if image is None:
            print("[ERROR] Could not download image. Check the URL and network.")
            sys.exit(1)
        image_label = args.url.split("/")[-1] or "downloaded_image"

    # ── Load model ─────────────────────────────────────────────────────
    ckpt = args.checkpoint or CKPT_PATH_V3
    if not os.path.exists(ckpt):
        print(f"[ERROR] Checkpoint not found: {ckpt}")
        print("Run: python train_transformer.py")
        sys.exit(1)

    scorer = AestheticScorerV3(
        checkpoint_path=ckpt,
        aesthetic_names=list(AESTHETIC_NAMES.keys()),
    )

    print(f"\nImage: {image_label}")
    print(f"Size : {image.size[0]}×{image.size[1]} px")
    print(f"Model : V3 (CLIP-free)")

    # ── Score ──────────────────────────────────────────────────────────
    if args.aesthetic:
        result = scorer.score_image(image, args.aesthetic)
        print_result(result)
    else:
        print("\nRanking against all aesthetics …")
        ranked = scorer.rank_aesthetics(image)
        print_ranking(ranked)


if __name__ == "__main__":
    main()
