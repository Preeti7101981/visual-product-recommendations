"""
Output Functions
=================
(1) cosine_similarity_score: cosine similarity between two embeddings.
(2) recommend_similar_images: given any input image, returns the top-N most
    visually similar images from the catalogue, using (1), excluding the
    query image itself if it happens to already be in the catalogue.

Embeddings are stored with filenames only (not full paths) - see
build_embeddings.py - so callers must supply which image directory those
filenames resolve against (the local cleaned catalogue during evaluation,
or the hosted catalogue directory in the deployed app).
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import config
from src.feature_creation import EmbeddingModel, get_embedding

_MODEL = None
_EMBEDDINGS = None
_FILENAMES = None


def _get_model() -> EmbeddingModel:
    global _MODEL
    if _MODEL is None:
        _MODEL = EmbeddingModel()
    return _MODEL


def load_embeddings(embeddings_path: str = None):
    global _EMBEDDINGS, _FILENAMES
    data = torch.load(embeddings_path or config.EMBEDDINGS_PATH)
    _EMBEDDINGS = data["embeddings"]
    _FILENAMES = data["filenames"]
    return _EMBEDDINGS, _FILENAMES


def cosine_similarity_score(embedding_a: torch.Tensor, embedding_b: torch.Tensor) -> float:
    """Cosine similarity between two L2-normalized embeddings (a simple dot product)."""
    return float(torch.dot(embedding_a, embedding_b))


def recommend_similar_images(
    input_image_path: str,
    image_dir: str,
    top_n: int = None,
    embeddings_path: str = None,
    exclude_filename: str = None,
):
    """Return the top_n most visually similar catalogue images to input_image_path.

    Args:
        input_image_path: path to the query image (can be anywhere on disk).
        image_dir: directory the catalogue filenames resolve against.
        top_n: how many recommendations to return (defaults to config.TOP_N).
        embeddings_path: defaults to config.EMBEDDINGS_PATH.
        exclude_filename: if the query image is itself a catalogue file
            (e.g. during evaluation), pass its filename here so it's excluded
            from its own recommendations (Issue 5).

    Returns: list of (full_image_path, score) tuples, sorted by descending score.
    """
    global _EMBEDDINGS, _FILENAMES
    top_n = top_n or config.TOP_N
    if _EMBEDDINGS is None:
        load_embeddings(embeddings_path)

    model = _get_model()
    query_embedding = get_embedding(model, input_image_path)

    scores = _EMBEDDINGS @ query_embedding

    if exclude_filename is not None:
        exclude_idx = [i for i, f in enumerate(_FILENAMES) if f == exclude_filename]
        for i in exclude_idx:
            scores[i] = -float("inf")  # guarantees it's never selected as a top match

    k = min(top_n, len(_FILENAMES))
    top_scores, top_indices = torch.topk(scores, k=k)

    results = [
        (str(Path(image_dir) / _FILENAMES[i]), float(top_scores[j]))
        for j, i in enumerate(top_indices)
    ]
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("query_image")
    parser.add_argument("--image_dir", default=config.DATA_PROCESSED_DIR)
    args = parser.parse_args()

    results = recommend_similar_images(args.query_image, image_dir=args.image_dir)
    for path, score in results:
        print(f"{path}: cosine similarity = {score:.4f}")
