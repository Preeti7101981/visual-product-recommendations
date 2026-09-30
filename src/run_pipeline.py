"""
Run Full Pipeline
==================
Runs the complete pipeline end-to-end, from a clean checkout, with a single
documented command:

    python src/run_pipeline.py

Steps:
    Dataset (already downloaded by you - see README) -> Cleaning ->
    Feature extraction & embedding generation -> Evaluation -> Results

This script creates every required directory itself, so nothing needs to be
manually prepared beyond placing the raw Kaggle images in data/raw/.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import config
from src import data_processing, build_embeddings, evaluate


def main():
    print("=" * 70)
    print("VISUAL PRODUCT RECOMMENDATION - FULL PIPELINE")
    print("=" * 70)

    config.ensure_required_dirs()

    raw_count = sum(
        1 for p in Path(config.DATA_RAW_DIR).rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    )
    if raw_count == 0:
        print(
            f"\nERROR: No images found in {config.DATA_RAW_DIR}\n"
            "Download the dataset first - see README.md 'Dataset setup'.\n"
        )
        sys.exit(1)

    print(f"\n[1/3] Cleaning data ({raw_count} raw images found)...")
    t0 = time.time()
    data_processing.process_images()
    print(f"Step 1 done in {time.time() - t0:.1f}s")

    print("\n[2/3] Extracting features and building embeddings...")
    t0 = time.time()
    build_embeddings.build_embeddings()
    print(f"Step 2 done in {time.time() - t0:.1f}s")

    print("\n[3/3] Running evaluation...")
    t0 = time.time()
    evaluate.run_evaluation()
    print(f"Step 3 done in {time.time() - t0:.1f}s")

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print(f"Embeddings:  {config.EMBEDDINGS_PATH}")
    print(f"Results:     {config.RECOMMENDATION_RESULTS_JSON}")
    print(f"Figures:     {config.RESULTS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
