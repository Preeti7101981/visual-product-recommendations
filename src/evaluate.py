"""
Evaluation
==========
Randomly selects config.N_EVAL_IMAGES images from the cleaned catalogue as
evaluation queries (using config.RANDOM_SEED, so this is reproducible without
any manual/CLI seed argument). For each query, generates the top-N
recommendations using the SAME model and embeddings as the deployed app,
excluding the query image itself from its own results. Saves a plotted
figure per query and a combined JSON summary under artifacts/results/.
"""

import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import config
from src.recommend import load_embeddings, recommend_similar_images

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def select_evaluation_queries():
    """Randomly select N_EVAL_IMAGES filenames from the cleaned catalogue,
    using the seed from config (not hardcoded here, not passed via CLI)."""
    _, filenames = load_embeddings(config.EMBEDDINGS_PATH)
    rng = random.Random(config.RANDOM_SEED)
    n = min(config.N_EVAL_IMAGES, len(filenames))
    return rng.sample(filenames, n)


def plot_query_result(query_path: str, results, save_path: str):
    n_results = len(results)
    fig, axes = plt.subplots(1, n_results + 1, figsize=(3 * (n_results + 1), 4))

    with Image.open(query_path) as img:
        axes[0].imshow(img.convert("RGB"))
    axes[0].set_title("Query image")
    axes[0].axis("off")

    for ax, (path, score) in zip(axes[1:], results):
        with Image.open(path) as img:
            ax.imshow(img.convert("RGB"))
        ax.set_title(f"Score: {score:.3f}")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_evaluation():
    config.ensure_required_dirs()
    query_filenames = select_evaluation_queries()
    print(f"Evaluation queries (seed={config.RANDOM_SEED}): {query_filenames}")

    all_results = []
    below_target = []

    for i, filename in enumerate(query_filenames, start=1):
        query_path = str(Path(config.DATA_PROCESSED_DIR) / filename)

        results = recommend_similar_images(
            query_path,
            image_dir=config.DATA_PROCESSED_DIR,
            top_n=config.TOP_N,
            exclude_filename=filename,  # never recommend the query image to itself
        )

        figure_path = str(Path(config.RESULTS_DIR) / f"eval_query_{i}_{Path(filename).stem}.png")
        plot_query_result(query_path, results, figure_path)

        entry = {
            "query_filename": filename,
            "recommendations": [
                {
                    "filename": str(Path(p).relative_to(config.DATA_PROCESSED_DIR).as_posix()),
                    "cosine_similarity": round(s, 4),
                }
                for p, s in results
            ],
            "figure": figure_path,
        }
        all_results.append(entry)

        for rec in entry["recommendations"]:
            if rec["cosine_similarity"] < config.TARGET_SIMILARITY:
                below_target.append((filename, rec["filename"], rec["cosine_similarity"]))

        print(f"\nQuery {i}: {filename}")
        for rec in entry["recommendations"]:
            print(f"  {rec['filename']}: {rec['cosine_similarity']}")

    summary = {
        "random_seed": config.RANDOM_SEED,
        "n_eval_images": config.N_EVAL_IMAGES,
        "top_n": config.TOP_N,
        "target_similarity": config.TARGET_SIMILARITY,
        "model": config.MODEL_NAME,
        "results": all_results,
    }
    with open(config.RECOMMENDATION_RESULTS_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved evaluation results -> {config.RECOMMENDATION_RESULTS_JSON}")
    if below_target:
        print(f"\nNote: {len(below_target)} recommendation(s) scored below the "
              f"{config.TARGET_SIMILARITY} target:")
        for q, r, s in below_target:
            print(f"  query={q} rec={r} score={s}")
    else:
        print(f"\nAll recommendations scored at or above the {config.TARGET_SIMILARITY} target.")


def main():
    run_evaluation()


if __name__ == "__main__":
    main()
