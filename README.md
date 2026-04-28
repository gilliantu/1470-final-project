# Outfit Aesthetic Scorer

**Team JAG** — gtu3, jshou, ajacob39  
CSCI 1470 Final Project

---

## Overview

This project rates how closely an outfit photo matches a fashion aesthetic (streetwear, cottagecore, dark academia, etc.) and recommends which individual clothing item best complements a base outfit.

Given an image, the model scores it 0–10 against ten aesthetics using a custom encoder-only transformer built on a frozen ResNet-50 backbone. Scores are within-image z-score normalized so the spread across aesthetics is meaningful for any photo.

**Supported aesthetics:** Y2K · Streetwear · Cottagecore · Dark Academia · Minimalist · Preppy · Boho · Grunge · Formal · Casual

---

## Architecture

The pipeline has two stages:

**1. Feature extraction — `PretrainedPatchEmbedder`**  
A frozen ResNet-50 (ImageNet weights) extracts multi-scale features from each image. Layer 3 (mid-level textures) and Layer 4 (high-level semantics) are fused via a 1×1 conv and reshaped into 49 patch tokens plus a learned CLS token → `[B, 50, 768]`.

**2. Classification — `AestheticTransformerV2`**  
An encoder-only transformer (6 layers, 8 heads, d_model=384) classifies from the CLS token. No positional encoding is used since CNN patch tokens already carry spatial information in their feature values. Output: one logit per aesthetic, z-score normalized to 0–10 at inference.

**Training** uses CLIP cosine similarities as soft labels (CLIP is only used to generate labels, not at inference). Loss = MSE + 0.5 × Pearson correlation, which jointly optimizes absolute scores and relative aesthetic rankings.

---

## Project structure

```
.
├── aesthetics.py          # Aesthetic definitions (10 styles, text prompts)
├── model.py               # CLIP-based scorer used by the web app
├── transformer_model.py   # ResNet-50 + transformer scorer used by the CLI
├── preprocess.py          # Data utilities for fashion.json dataset
├── train.py               # Build CLIP prototype embeddings (data/prototypes.npz)
├── train_transformer.py   # Train the V3 transformer (data/transformer_v3.pt)
├── test.py                # Evaluate the CLIP-based scorer on fashion.json
├── app.py                 # Flask web app (scoring + item recommendation)
├── main.py                # CLI (scoring + item recommendation)
├── templates/index.html   # Web UI
├── data/
│   ├── prototypes.npz     # CLIP prototype embeddings (used by web app)
│   └── transformer_v3.pt  # Trained transformer checkpoint (used by CLI)
└── fashion.json/          # Dataset directory (fashion.json + fashion-cat.json)
```

---

## Setup

```bash
conda activate csci1470
pip install -r requirements.txt
```

The trained model files (`data/prototypes.npz` and `data/transformer_v3.pt`) are included. If you need to retrain from scratch, see the Training section below.

---

## Running the web app

```bash
python app.py
```

Open **http://localhost:5001** in your browser.

### Score Outfit tab
Upload an outfit photo and optionally select a target aesthetic. Click **Score** to get:
- A 0–10 score for every aesthetic, ranked from best to worst
- A progress bar visualization for each aesthetic

### Find Best Match tab
Upload a **base outfit** on the left and one or more **individual clothing items** on the right (drag-and-drop or click to browse, multiple files supported). Click **Find Best Match** to get:
- The base outfit's detected top aesthetic
- Each item's detected aesthetic and its compatibility score with the base outfit's aesthetic
- A recommendation for the single best-matching item, highlighted with a ⭐

---

## CLI

```bash
# List all available aesthetics
python main.py --list

# Score an outfit against all aesthetics (ranked)
python main.py --image outfit.jpg

# Score against a specific aesthetic
python main.py --image outfit.jpg --aesthetic streetwear

# Score from a URL
python main.py --url <image_url>

# Recommend which item best matches a base outfit
python main.py --recommend \
  --base outfit.jpg \
  --items item1.jpg item2.jpg item3.jpg
```

The `--recommend` mode prints:
1. The base outfit's top aesthetic and score
2. Each item's detected aesthetic with a `✓ MATCH` or `✗` tag
3. A recommendation box naming the best item, with the reasoning (exact match vs. closest match)

---

## Training from scratch

**Step 1 — Build CLIP prototype embeddings** (required for the web app and to generate training labels):
```bash
python train.py
# Optionally blend in image references from fashion.json:
python train.py --images 30 --blend
```

**Step 2 — Train the transformer** (required for the CLI):
```bash
python train_transformer.py
# Faster on subsequent runs using the cached features:
python train_transformer.py --use-cache
# Fine-tune the ResNet backbone too (slower, needs more data):
python train_transformer.py --fine-tune-backbone
```

**Evaluate the CLIP-based scorer on random fashion.json scenes:**
```bash
python test.py --n 50
python test.py --image outfit.jpg      # single local image
python test.py --n 50 --output results/eval.csv
```

---

## Dataset

[Shop The Look (Kaggle)](https://www.kaggle.com/datasets/pypiahmad/shop-the-look-dataset/data?select=fashion.json) — scene-product pairs from Pinterest. Used to download reference images for training and evaluation. The dataset files (`fashion.json/fashion.json` and `fashion.json/fashion-cat.json`) must be present locally for training; inference on your own photos requires no dataset.
