"""
Upload Full Catalogue to Hugging Face Hub
============================================
Uploads the complete cleaned catalogue (all images in data/processed/) and
the corresponding artifacts/embeddings.pt to a Hugging Face Hub dataset
repository, so the deployed app can serve the FULL catalogue rather than a
reduced subset (Issue 4).

One-time setup before running this:
    1. Create a free account at https://huggingface.co/join
    2. Create an access token (write access) at
       https://huggingface.co/settings/tokens
    3. Run: huggingface-cli login
       (paste your token when prompted)
    4. Set HF_DATASET_REPO_ID in config/config.py to your own
       "your-username/repo-name"

Then run:
    python src/upload_to_hf.py

This can take a while for ~4.3GB of images depending on your upload speed -
it's safe to leave running, and huggingface_hub resumes cleanly if
interrupted (just rerun the command).
"""

import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import config


def upload_catalogue():
    if "REPLACE_WITH_YOUR_HF_USERNAME" in config.HF_DATASET_REPO_ID:
        print(
            "ERROR: set HF_DATASET_REPO_ID in config/config.py to your own "
            "'your-username/repo-name' before running this script."
        )
        sys.exit(1)

    api = HfApi()

    print(f"Creating (or reusing) dataset repo: {config.HF_DATASET_REPO_ID}")
    create_repo(
        repo_id=config.HF_DATASET_REPO_ID,
        repo_type="dataset",
        exist_ok=True,
    )

    processed_dir = Path(config.DATA_PROCESSED_DIR)
    n_images = len(list(processed_dir.glob("*")))
    print(f"Uploading {n_images} images from {processed_dir} -> "
          f"{config.HF_DATASET_REPO_ID}/{config.HF_IMAGES_SUBDIR}/ ...")
    print("(This uploads the whole folder in one call and can take a while for ~4GB+.)")

    api.upload_folder(
        folder_path=str(processed_dir),
        path_in_repo=config.HF_IMAGES_SUBDIR,
        repo_id=config.HF_DATASET_REPO_ID,
        repo_type="dataset",
    )
    print("Images uploaded.")

    print(f"Uploading embeddings file: {config.EMBEDDINGS_PATH}")
    api.upload_file(
        path_or_fileobj=config.EMBEDDINGS_PATH,
        path_in_repo=config.HF_EMBEDDINGS_FILENAME,
        repo_id=config.HF_DATASET_REPO_ID,
        repo_type="dataset",
    )
    print("Embeddings uploaded.")

    print(f"\nDone. Full catalogue is hosted at: "
          f"https://huggingface.co/datasets/{config.HF_DATASET_REPO_ID}")


if __name__ == "__main__":
    upload_catalogue()
