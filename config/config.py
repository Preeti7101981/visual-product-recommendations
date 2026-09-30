"""
Central configuration for the Visual Product Recommendation project.

All paths, the chosen model, and evaluation settings (including the random
seed) live here rather than being hardcoded in individual scripts or passed
via command-line arguments. This is the single source of truth every script
imports from.
"""

import os

# ---------------------------------------------------------------------------
# Final model (Issue 3: one consistent model used everywhere)
# ---------------------------------------------------------------------------
# ResNet50 is the single feature-extraction model used for embedding
# generation, recommendation, evaluation, AND deployment. There is no
# separate "lite" model anywhere in this project.
MODEL_NAME = "resnet50"
EMBEDDING_DIM = 2048

# ---------------------------------------------------------------------------
# Evaluation settings (Issue 5)
# ---------------------------------------------------------------------------
# Fixed here, not hardcoded in scripts and not passed via CLI arguments, so
# the evaluation is reproducible by anyone running run_pipeline.py.
RANDOM_SEED = 42
N_EVAL_IMAGES = 3       # number of randomly sampled catalogue images to use as evaluation queries
TOP_N = 5               # number of recommendations returned per query
TARGET_SIMILARITY = 0.7 # sanity-check threshold referenced in documentation/evaluation output

# ---------------------------------------------------------------------------
# Paths (all relative to the project root; created automatically by
# run_pipeline.py so a clean checkout works with no manual folder setup)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DATA_RAW_DIR = os.path.join(DATA_DIR, "raw")
DATA_PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
DATA_TEST_IMAGES_DIR = os.path.join(DATA_DIR, "test", "images")
CLEANED_MANIFEST_CSV = os.path.join(DATA_DIR, "cleaned_manifest.csv")

ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
EMBEDDINGS_PATH = os.path.join(ARTIFACTS_DIR, "embeddings.pt")
RESULTS_DIR = os.path.join(ARTIFACTS_DIR, "results")
RECOMMENDATION_RESULTS_JSON = os.path.join(RESULTS_DIR, "recommendation_results.json")

UPLOAD_DIR = os.path.join(PROJECT_ROOT, "static", "uploads")

# Every directory that must exist before the pipeline runs. run_pipeline.py
# creates these automatically so the project works from a clean checkout.
REQUIRED_DIRS = [
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    DATA_TEST_IMAGES_DIR,
    ARTIFACTS_DIR,
    RESULTS_DIR,
    UPLOAD_DIR,
]

# ---------------------------------------------------------------------------
# Hugging Face Hub hosting (Issue 4: deploy the FULL cleaned catalogue)
# ---------------------------------------------------------------------------
# The full cleaned catalogue (~31,702 images, ~4.3GB) is hosted as a Hugging
# Face Hub dataset repo rather than a reduced subset, since it's too large
# to practically store in this Git repository. app.py downloads it from
# here at startup if it isn't already present locally.
HF_DATASET_REPO_ID = "Preeti7101981/visual-product-recommendations-catalogue"
HF_EMBEDDINGS_FILENAME = "embeddings.pt"
HF_IMAGES_SUBDIR = "images"


def ensure_required_dirs():
    """Create every directory the pipeline needs, if it doesn't already exist."""
    for d in REQUIRED_DIRS:
        os.makedirs(d, exist_ok=True)
