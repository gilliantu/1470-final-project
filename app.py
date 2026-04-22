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


if __name__ == "__main__":
    app.run(debug=False, port=5001)
