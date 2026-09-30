"""
Model Deployment
================
Flask web app. Uses the SAME model, SAME embeddings-generation code, and the
FULL cleaned catalogue as the rest of this project (Issues 3 and 4) - no
separate lite model, no reduced subset.

On startup, if the full catalogue + embeddings aren't already present
locally (e.g. in a fresh container), they're downloaded once from the
Hugging Face Hub dataset repo configured in config.HF_DATASET_REPO_ID.

Run locally:
    python src/app.py
Then visit http://localhost:5000
"""

import os
import sys
import uuid
from pathlib import Path

from flask import Flask, render_template, request, send_from_directory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import config
from src.recommend import load_embeddings, recommend_similar_images

app = Flask(__name__, template_folder=str(Path(__file__).resolve().parent.parent / "templates"))

# Where the downloaded full catalogue + embeddings live inside the deployed
# container. Kept separate from data/processed (the local pipeline output)
# so a deployment container never needs the raw Kaggle download.
DEPLOY_CACHE_DIR = Path(os.environ.get(
    "DEPLOY_CACHE_DIR",
    str(Path(__file__).resolve().parent.parent / ".deploy_cache")
))
DEPLOY_IMAGES_DIR = DEPLOY_CACHE_DIR / config.HF_IMAGES_SUBDIR
DEPLOY_EMBEDDINGS_PATH = DEPLOY_CACHE_DIR / config.HF_EMBEDDINGS_FILENAME

app.config["UPLOAD_FOLDER"] = config.UPLOAD_DIR
os.makedirs(config.UPLOAD_DIR, exist_ok=True)


def ensure_catalogue_downloaded():
    """Downloads the full catalogue + embeddings from Hugging Face Hub if not
    already cached locally. Runs once per fresh container."""
    if DEPLOY_EMBEDDINGS_PATH.exists() and DEPLOY_IMAGES_DIR.exists():
        print(f"Catalogue already cached at {DEPLOY_CACHE_DIR}, skipping download.")
        return

    from huggingface_hub import snapshot_download

    print(f"Downloading full catalogue from {config.HF_DATASET_REPO_ID} "
          f"-> {DEPLOY_CACHE_DIR} (first run only, this can take a while)...")
    DEPLOY_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=config.HF_DATASET_REPO_ID,
        repo_type="dataset",
        local_dir=str(DEPLOY_CACHE_DIR),
    )
    print("Catalogue download complete.")


ensure_catalogue_downloaded()
load_embeddings(str(DEPLOY_EMBEDDINGS_PATH))


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/catalogue_images/<path:filename>")
def catalogue_images(filename):
    return send_from_directory(str(DEPLOY_IMAGES_DIR), filename)


@app.route("/uploaded_images/<path:filename>")
def uploaded_images(filename):
    return send_from_directory(config.UPLOAD_DIR, filename)


@app.route("/recommend", methods=["POST"])
def recommend():
    if "image" not in request.files or request.files["image"].filename == "":
        return render_template("index.html", error="Please choose an image to upload.")

    file = request.files["image"]
    filename = f"{uuid.uuid4().hex}_{file.filename}"
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)

    results = recommend_similar_images(
        save_path,
        image_dir=str(DEPLOY_IMAGES_DIR),
        top_n=config.TOP_N,
        embeddings_path=str(DEPLOY_EMBEDDINGS_PATH),
    )

    recommendations = [
        {
            "image_url": f"/catalogue_images/{Path(path).relative_to(DEPLOY_IMAGES_DIR).as_posix()}",
            "score": f"{score:.3f}",
        }
        for path, score in results
    ]

    return render_template(
        "index.html",
        uploaded_image=f"/uploaded_images/{filename}",
        recommendations=recommendations,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
