"""
Clean Up Stray Files on Hugging Face Hub
===========================================
The first upload attempt (before sharding) partially succeeded before
hitting the "too many files per directory" error - meaning some loose
files were committed directly under images/ (not inside a shard_XX
subfolder) before the run failed. This script finds and deletes exactly
those stray files, leaving the properly sharded structure untouched.

Usage:
    python cleanup_hf_stray_files.py
"""

import sys
from pathlib import Path

from huggingface_hub import HfApi, CommitOperationDelete

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import config


def cleanup():
    api = HfApi()
    all_files = api.list_repo_files(repo_id=config.HF_DATASET_REPO_ID, repo_type="dataset")

    stray_files = [
        f for f in all_files
        if f.startswith("images/")
        and f.count("/") == 1  # "images/xxx.jpg" - one slash = direct child, not in a shard folder
    ]

    print(f"Found {len(all_files)} total files in repo.")
    print(f"Found {len(stray_files)} stray loose files to remove (leftover from the failed first attempt).")

    if not stray_files:
        print("Nothing to clean up.")
        return

    # Delete in batches to keep each commit a reasonable size
    batch_size = 500
    for i in range(0, len(stray_files), batch_size):
        batch = stray_files[i:i + batch_size]
        operations = [CommitOperationDelete(path_in_repo=f) for f in batch]
        api.create_commit(
            repo_id=config.HF_DATASET_REPO_ID,
            repo_type="dataset",
            operations=operations,
            commit_message=f"Remove {len(batch)} stray loose files from failed initial upload",
        )
        print(f"  Deleted batch {i // batch_size + 1}: {len(batch)} files")

    print(f"\nDone. Removed {len(stray_files)} stray files.")


if __name__ == "__main__":
    cleanup()
