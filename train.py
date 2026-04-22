"""
train.py — Build and save aesthetic prototype embeddings.

Run this once before using main.py / test.py:

    conda activate csci1470
    python train.py [--images N] [--blend]

What it does
------------
1.  For every aesthetic defined in aesthetics.py, encodes the text prompts
    with CLIP and forms a text prototype.
2.  Optionally downloads N scene images per aesthetic from fashion.json
    (using product-category hints) and blends image features in.
3.  Saves all prototypes to data/prototypes.npz.
"""

import argparse
import os
import sys

import numpy as np

from aesthetics import AESTHETICS, AESTHETIC_NAMES
from model import AestheticScorer, save_prototypes
from preprocess import (
    load_records,
    load_categories,
    sample_scenes_for_categories,
    batch_download,
    CATEGORY_AESTHETIC_MAP,
)

PROTO_PATH = os.path.join(os.path.dirname(__file__), "data", "prototypes.npz")

# Map each aesthetic to relevant product category keywords for seeding images
AESTHETIC_CAT_HINTS = {
    "y2k":           ["Shirts & Tops", "Pants", "Dresses"],
    "streetwear":    ["Shirts & Tops", "Jackets", "Shorts"],
    "cottagecore":   ["Dresses", "Skirts"],
    "dark_academia": ["Outerwear", "Coats", "Shirts & Tops"],
    "minimalist":    ["Shirts & Tops", "Pants"],
    "preppy":        ["Shirts & Tops", "Shorts", "Skirts"],
    "boho":          ["Dresses", "Skirts"],
    "grunge":        ["Outerwear", "Jackets", "Shirts & Tops"],
    "formal":        ["Suits", "Shirts & Tops"],
    "casual":        ["Shirts & Tops", "Pants", "Shorts"],
}


def build_prototypes(
    scorer: AestheticScorer,
    n_images: int = 0,
    blend: bool = False,
    text_weight: float = 0.6,
) -> dict[str, np.ndarray]:
    """
    Build prototype embeddings for every aesthetic.

    Parameters
    ----------
    scorer      : AestheticScorer instance
    n_images    : number of reference scene images to download per aesthetic
                  (0 = text-only, no downloads)
    blend       : whether to blend text and image prototypes
    text_weight : fraction of text prototype when blending (0-1)
    """
    records = categories = None
    if n_images > 0:
        print("Loading fashion.json dataset …")
        records = load_records()
        categories = load_categories()

    prototypes = {}

    for aes_key, prompts in AESTHETICS.items():
        display = AESTHETIC_NAMES.get(aes_key, aes_key)
        print(f"\n── {display} ──")

        # 1. Text prototype (always computed)
        print(f"   Encoding {len(prompts)} text prompts …")
        text_proto = scorer.build_text_prototype(prompts)

        # 2. Image prototype (optional)
        image_proto = None
        if n_images > 0 and records is not None:
            cat_hints = AESTHETIC_CAT_HINTS.get(aes_key, ["Shirts & Tops"])
            urls = sample_scenes_for_categories(
                records, categories, cat_hints, n=n_images
            )
            print(f"   Downloading {len(urls)} scene images …")
            images = batch_download(urls, verbose=True)
            if images:
                print(f"   Downloaded {len(images)}/{len(urls)} images.")
                image_proto = scorer.build_image_prototype(images)
            else:
                print("   No images downloaded; falling back to text-only.")

        # 3. Blend or use text-only
        if blend and image_proto is not None:
            proto = scorer.blend_prototypes(text_proto, image_proto, text_weight)
            print(f"   Blended (text {text_weight:.0%} / image {1-text_weight:.0%})")
        else:
            proto = text_proto
            if n_images > 0:
                print("   Using text-only prototype (no --blend flag).")

        prototypes[aes_key] = proto

    return prototypes


def main():
    parser = argparse.ArgumentParser(description="Build aesthetic prototype embeddings.")
    parser.add_argument(
        "--images", type=int, default=0,
        help="Number of fashion.json scene images to download per aesthetic (default 0 = text-only).",
    )
    parser.add_argument(
        "--blend", action="store_true",
        help="Blend text and image prototypes (requires --images > 0).",
    )
    parser.add_argument(
        "--text-weight", type=float, default=0.6,
        help="Weight for text prototype when blending (default 0.6).",
    )
    parser.add_argument(
        "--output", type=str, default=PROTO_PATH,
        help=f"Output path for prototypes .npz file (default {PROTO_PATH}).",
    )
    args = parser.parse_args()

    scorer = AestheticScorer()
    prototypes = build_prototypes(
        scorer,
        n_images=args.images,
        blend=args.blend,
        text_weight=args.text_weight,
    )
    save_prototypes(prototypes, args.output)
    print(f"\nDone. {len(prototypes)} prototypes saved to {args.output}")


if __name__ == "__main__":
    main()
