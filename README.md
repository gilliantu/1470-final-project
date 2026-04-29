# Outfit Aesthetic Scorer

**Team JAG** — gtu3, jshou, ajacob39  
CSCI 1470 Final Project

---

## Overview

This project rates how closely an outfit photo matches a fashion aesthetic and recommends which individual clothing item best complements a base outfit. Given a photo, the model scores it 0–10 against ten aesthetics using a custom encoder-only transformer built on a frozen ResNet-50 backbone. Scores are within-image z-score normalized so the spread across aesthetics is meaningful for any photo.

**Supported aesthetics:** Y2K · Streetwear · Cottagecore · Dark Academia · Minimalist · Preppy · Boho · Grunge · Formal · Casual

---

## Architecture

The model is fully CLIP-free at inference and runs in two stages:

**1. Feature extraction — `PretrainedPatchEmbedder`**  
A frozen ResNet-50 (ImageNet weights) extracts multi-scale features. Layer 3 (mid-level textures, `[1024, 14, 14]`) and Layer 4 (high-level semantics, `[2048, 7, 7]`) are fused via a 1×1 conv, reshaped into 49 spatial patch tokens, and prepended with a learned CLS token → `[B, 50, 768]`.

**2. Classification — `AestheticTransformerV2`**  
An encoder-only transformer (6 layers, 8 heads, d_model=384) processes the patch tokens and classifies from the CLS token. Pre-LayerNorm and GELU activation are used throughout. No positional encoding is applied — CNN patch tokens already encode spatial information in their feature values. Output: one logit per aesthetic, z-score normalized to 0–10 at inference.

**Training** uses CLIP cosine similarities as soft labels — CLIP encodes training images and computes similarity against hand-crafted aesthetic prototype vectors, producing regression targets. CLIP is only used to generate these labels and is not present at inference. Loss = MSE + 0.5 × Pearson correlation, jointly optimizing absolute scores and relative aesthetic rankings.

---

## Project Structure

```
.
├── aesthetics.py          # 10 aesthetic definitions with text prompts and display names
├── model.py               # CLIP-based scorer (used for training label generation)
├── transformer_model.py   # ResNet-50 + transformer scorer (used for inference)
├── preprocess.py          # Data utilities for the fashion.json dataset
├── train.py               # Build CLIP prototype embeddings → data/prototypes.npz
├── train_transformer.py   # Train the V3 transformer → data/transformer_v3.pt
├── test.py                # Evaluate the CLIP prototype scorer on fashion.json images
├── validate.py            # Quantitative comparison: V3 transformer vs. CLIP baseline
├── app.py                 # Flask web app (scoring + item recommendation, uses V3)
├── main.py                # CLI (scoring + item recommendation, uses V3)
├── templates/index.html   # Web UI
├── data/
│   ├── prototypes.npz     # CLIP prototype embeddings (used for training labels)
│   └── transformer_v3.pt  # Trained transformer checkpoint (used for inference)
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

## Running the Web App

```bash
python app.py
```

Open **http://localhost:5001** in your browser.

**Score Outfit tab** — upload an outfit photo and optionally pick a target aesthetic. Returns a 0–10 score for every aesthetic ranked from best to worst with animated bar visualizations.

**Find Best Match tab** — upload a base outfit on the left and one or more individual clothing items on the right (drag-and-drop or multi-file select supported). Returns the base outfit's detected aesthetic, each item's detected aesthetic and compatibility score, and a ⭐ recommendation for the best-matching item.

---

## CLI

```bash
# List all available aesthetics
python main.py --list

# Score an outfit against all aesthetics (ranked)
python main.py --image outfit.jpg

# Score against one specific aesthetic
python main.py --image outfit.jpg --aesthetic streetwear

# Score from a URL
python main.py --url <image_url>

# Recommend which item best matches a base outfit
python main.py --recommend \
  --base outfit.jpg \
  --items item1.jpg item2.jpg item3.jpg
```

The `--recommend` mode prints each item's detected aesthetic with a `✓ MATCH` or `✗` tag and a final recommendation box with reasoning (exact aesthetic match vs. closest match).

---

## Validation

Compare the V3 transformer against the CLIP prototype baseline across a set of test images:

```bash
# Run on personal_photos_for_testing/ (default)
python validate.py

# Download N images from fashion.json instead
python validate.py --download 50
```

Prints model agreement (Spearman rank correlation, top-1 and top-3 agreement rates), confidence margin, per-aesthetic score distributions, and per-image predictions from both models.

---

## Training from Scratch

**Step 1 — Build CLIP prototype embeddings** (required to generate training labels):
```bash
python train.py

# Optionally blend image references from fashion.json into the prototypes:
python train.py --images 30 --blend
```

**Step 2 — Train the V3 transformer:**
```bash
python train_transformer.py

# Faster on subsequent runs using cached ResNet features:
python train_transformer.py --use-cache

# Also fine-tune ResNet layer3 + layer4 (slower, needs more data):
python train_transformer.py --fine-tune-backbone
```

**Dependency order:**
```
train.py → train_transformer.py → app.py / main.py / validate.py
```

**Evaluate the CLIP prototype scorer on fashion.json scenes:**
```bash
python test.py --n 50
python test.py --image outfit.jpg
python test.py --n 50 --output results/eval.csv
```

---

## Report

See [`report.pdf`](report.pdf) for the full written report.

---

## Dataset

[Shop The Look (Kaggle)](https://www.kaggle.com/datasets/pypiahmad/shop-the-look-dataset/data?select=fashion.json) — scene-product image pairs from Pinterest. The dataset files (`fashion.json/fashion.json` and `fashion.json/fashion-cat.json`) are required for training; inference on your own photos requires no dataset.
