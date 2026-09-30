"""
Build and Save Embeddings for the Full Cleaned Catalogue
==========================================================
(1) Loads the final embedding model (ResNet50, see feature_creation.py).
(2) Applies it to every image in the cleaned catalogue.
(3) Saves the resulting embeddings + filenames to artifacts/embeddings.pt.

Run standalone:
    python src/build_embeddings.py
or as part of the full pipeline:
    python src/run_pipeline.py
"""

import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import config
from src.feature_creation import EmbeddingModel, get_embedding

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def build_embeddings(cleaned_dir: str = None, out_path: str = None):
    cleaned_dir = Path(cleaned_dir or config.DATA_PROCESSED_DIR)
    out_path = out_path or config.EMBEDDINGS_PATH

    image_paths = sorted(
        p for p in cleaned_dir.iterdir()
        if p.suffix.lower() in VALID_EXTENSIONS and p.is_file()
    )
    print(f"Building embeddings for {len(image_paths)} images from {cleaned_dir}...")

    if not image_paths:
        print(f"WARNING: no images found in {cleaned_dir}. Run data_processing.py first.")
        return

    model = EmbeddingModel()

    embeddings = []
    filenames = []
    for path in tqdm(image_paths, desc="Embedding", unit="img"):
        try:
            emb = get_embedding(model, str(path))
        except Exception as e:
            print(f"Skipping {path.name}: {e}")
            continue
        embeddings.append(emb)
        # Store filenames as relative, forward-slash paths so the mapping is
        # portable across machines / OSes (Issue 4: correct filename mapping)
        filenames.append(path.name)

    embeddings_tensor = torch.stack(embeddings)
    torch.save({"embeddings": embeddings_tensor, "filenames": filenames}, out_path)
    print(f"Saved {embeddings_tensor.shape[0]} embeddings of dim {embeddings_tensor.shape[1]} -> {out_path}")


def main():
    config.ensure_required_dirs()
    build_embeddings()


if __name__ == "__main__":
    main()
