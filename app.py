"""
app.py — Flask web UI for the outfit aesthetic scorer.

Run:
    conda activate csci1470
    python app.py

Then open http://localhost:5000
"""

import base64
import io
import os
import sys

from flask import Flask, jsonify, render_template, request
from PIL import Image

from aesthetics import AESTHETIC_NAMES
from model import AestheticScorer, load_prototypes

PROTO_PATH = os.path.join(os.path.dirname(__file__), "data", "prototypes.npz")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload limit

# Load model + prototypes once at startup
print("Loading prototypes…")
if not os.path.exists(PROTO_PATH):
    print(f"ERROR: {PROTO_PATH} not found. Run `python train.py` first.")
    sys.exit(1)

prototypes = load_prototypes(PROTO_PATH)
scorer = AestheticScorer()
print("Ready.")


@app.route("/")
def index():
    aesthetics = [
        {"key": k, "label": AESTHETIC_NAMES.get(k, k)}
        for k in prototypes
    ]
    return render_template("index.html", aesthetics=aesthetics)


@app.route("/score", methods=["POST"])
def score():
    # Accept either a file upload or a base64 data-URL from the browser
    image = None

    if "image" in request.files and request.files["image"].filename:
        f = request.files["image"]
        image = Image.open(f.stream).convert("RGB")
    elif request.is_json:
        data = request.get_json()
        if data and "image_b64" in data:
            b64 = data["image_b64"].split(",")[-1]  # strip data:...;base64,
            image = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")

    if image is None:
        return jsonify({"error": "No image provided."}), 400

    target = request.form.get("aesthetic") or (
        request.get_json(silent=True) or {}
    ).get("aesthetic")

    ranked = scorer.rank_aesthetics(image, prototypes)

    # Attach display labels
    for r in ranked:
        r["label_display"] = AESTHETIC_NAMES.get(r["aesthetic"], r["aesthetic"])

    if target and target in prototypes:
        for r in ranked:
            if r["aesthetic"] == target:
                return jsonify({"mode": "single", "result": r, "all": ranked})
        return jsonify({"error": f"Unknown aesthetic: {target}"}), 400

    return jsonify({"mode": "all", "all": ranked})


@app.route("/recommend", methods=["POST"])
def recommend():
    # Base outfit image
    if "base" not in request.files or not request.files["base"].filename:
        return jsonify({"error": "No base outfit image provided."}), 400
    base_img = Image.open(request.files["base"].stream).convert("RGB")

    # Item images (multiple files under key "items")
    item_files = request.files.getlist("items")
    valid_items = [
        (f.filename, Image.open(f.stream).convert("RGB"))
        for f in item_files if f.filename
    ]
    if not valid_items:
        return jsonify({"error": "No item images provided."}), 400

    # Score base outfit
    base_ranked = scorer.rank_aesthetics(base_img, prototypes)
    for r in base_ranked:
        r["label_display"] = AESTHETIC_NAMES.get(r["aesthetic"], r["aesthetic"])
    base_top_aesthetic = base_ranked[0]["aesthetic"]

    # Score each item
    items_out = []
    for idx, (name, img) in enumerate(valid_items):
        ranked = scorer.rank_aesthetics(img, prototypes)
        for r in ranked:
            r["label_display"] = AESTHETIC_NAMES.get(r["aesthetic"], r["aesthetic"])
        item_top = ranked[0]
        base_aes_entry = next(
            (r for r in ranked if r["aesthetic"] == base_top_aesthetic), None
        )
        items_out.append({
            "index": idx,
            "name": name,
            "top_aesthetic": item_top["aesthetic"],
            "top_aesthetic_display": item_top["label_display"],
            "top_score": round(item_top["score"], 2),
            "base_aesthetic_score": round(base_aes_entry["score"], 2) if base_aes_entry else 0.0,
            "matches_base": item_top["aesthetic"] == base_top_aesthetic,
        })

    # Recommend: exact aesthetic match first, otherwise highest score for base aesthetic
    exact_matches = [it for it in items_out if it["matches_base"]]
    if exact_matches:
        recommended = max(exact_matches, key=lambda x: x["base_aesthetic_score"])
    else:
        recommended = max(items_out, key=lambda x: x["base_aesthetic_score"])

    return jsonify({
        "base_aesthetic": base_top_aesthetic,
        "base_aesthetic_display": AESTHETIC_NAMES.get(base_top_aesthetic, base_top_aesthetic),
        "base_top_score": round(base_ranked[0]["score"], 2),
        "items": items_out,
        "recommended_index": recommended["index"],
        "exact_match": len(exact_matches) > 0,
    })


if __name__ == "__main__":
    app.run(debug=False, port=5001)
