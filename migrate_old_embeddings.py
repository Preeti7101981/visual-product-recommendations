"""
Migrate Old Embeddings File
=============================
One-time helper: converts an embeddings.pt saved by the OLD project
structure (where "filenames" contains full paths like
"data\\cleaned_images\\xxxx.jpg") into the new format this project expects
(filenames only, e.g. "xxxx.jpg" - see build_embeddings.py).

This does NOT recompute any embeddings - it just fixes the stored filename
strings, so it runs in seconds instead of hours.

Usage:
    python migrate_old_embeddings.py old_embeddings.pt artifacts/embeddings.pt
"""

import sys
from pathlib import Path

import torch


def migrate(old_path: str, new_path: str):
    data = torch.load(old_path)
    embeddings = data["embeddings"]
    old_filenames = data["filenames"]

    new_filenames = [Path(f).name for f in old_filenames]

    Path(new_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"embeddings": embeddings, "filenames": new_filenames}, new_path)

    print(f"Migrated {len(new_filenames)} entries.")
    print(f"Example filename: '{old_filenames[0]}' -> '{new_filenames[0]}'")
    print(f"Saved -> {new_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python migrate_old_embeddings.py <old_embeddings.pt> <new_embeddings_path>")
        sys.exit(1)
    migrate(sys.argv[1], sys.argv[2])
