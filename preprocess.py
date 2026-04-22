"""
preprocess.py — Data utilities for the fashion.json dataset.

Loads scene-product pairs and product category mappings.
Provides helpers for converting Pinterest image signatures to URLs
and downloading images for building aesthetic prototypes.
"""

import json
import os
import random
import time
from io import BytesIO

import requests
from PIL import Image

DATA_DIR = os.path.join(os.path.dirname(__file__), "fashion.json")
FASHION_JSON = os.path.join(DATA_DIR, "fashion.json")
FASHION_CAT_JSON = os.path.join(DATA_DIR, "fashion-cat.json")

# Category keyword → aesthetic name mapping for seeding reference images
CATEGORY_AESTHETIC_MAP = {
    "Shirts & Tops": ["casual", "streetwear", "preppy"],
    "Pants": ["casual", "formal", "streetwear"],
    "Dresses": ["cottagecore", "boho", "formal"],
    "Shorts": ["casual", "preppy"],
    "Outerwear": ["grunge", "streetwear", "dark_academia"],
    "Coats": ["dark_academia", "formal", "minimalist"],
    "Jackets": ["streetwear", "grunge", "casual"],
    "Shoes": ["casual", "formal", "streetwear"],
    "Suits": ["formal"],
    "Skirts": ["cottagecore", "preppy", "boho"],
}


def convert_to_url(signature: str) -> str:
    """Convert a Pinterest image signature to a CDN URL."""
    prefix = "http://i.pinimg.com/400x/%s/%s/%s/%s.jpg"
    return prefix % (signature[0:2], signature[2:4], signature[4:6], signature)


def load_records(path: str = FASHION_JSON) -> list[dict]:
    """Load JSONL fashion data into a list of dicts."""
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_categories(path: str = FASHION_CAT_JSON) -> dict:
    """Load product ID → category string mapping."""
    with open(path, "r") as f:
        return json.load(f)


def download_image(url: str, timeout: int = 8) -> Image.Image | None:
    """Download and return a PIL Image, or None on failure."""
    try:
        resp = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        if resp.status_code == 200:
            return Image.open(BytesIO(resp.content)).convert("RGB")
    except Exception:
        pass
    return None


def sample_scenes_for_categories(
    records: list[dict],
    categories: dict,
    target_keywords: list[str],
    n: int = 30,
    seed: int = 42,
) -> list[str]:
    """
    Return up to `n` scene image URLs whose associated product falls in
    one of the provided category keywords.
    """
    rng = random.Random(seed)
    matching_scenes = set()
    for record in records:
        pid = record["product"]
        cat = categories.get(pid, "")
        if any(kw.lower() in cat.lower() for kw in target_keywords):
            matching_scenes.add(record["scene"])

    selected = rng.sample(list(matching_scenes), min(n, len(matching_scenes)))
    return [convert_to_url(sig) for sig in selected]


def sample_random_scenes(records: list[dict], n: int = 50, seed: int = 0) -> list[str]:
    """Return `n` random unique scene image URLs from the dataset."""
    rng = random.Random(seed)
    unique = list({r["scene"] for r in records})
    selected = rng.sample(unique, min(n, len(unique)))
    return [convert_to_url(sig) for sig in selected]


def batch_download(
    urls: list[str],
    delay: float = 0.1,
    verbose: bool = True,
) -> list[Image.Image]:
    """Download a list of URLs, returning successfully fetched PIL images."""
    images = []
    iterator = urls
    if verbose:
        try:
            from tqdm import tqdm
            iterator = tqdm(urls, desc="Downloading images")
        except ImportError:
            pass
    for url in iterator:
        img = download_image(url)
        if img is not None:
            images.append(img)
        time.sleep(delay)
    return images
