"""
Shard the Cleaned Catalogue
==============================
Hugging Face Hub repos reject more than 10,000 files in a single directory.
This script splits data/processed/ (currently one flat folder) into several
shard subfolders (shard_00, shard_01, ...), each under that limit, and
updates artifacts/embeddings.pt so its "filenames" list points at the new
sharded relative paths (e.g. "shard_00/abc123.jpg") instead of bare
filenames - keeping the filename-to-embedding-to-image mapping exactly
correct throughout.

This COPIES files into a new data/processed_sharded/ folder rather than
modifying data/processed/ in place, so nothing is destroyed if something
goes wrong. Once you've confirmed it worked, delete the old flat folder
yourself to reclaim disk space.

Usage:
    python shard_processed_images.py
"""

import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import config

SHARD_SIZE = 8000  # keeps each shard comfortably under the 10,000 limit


def shard_catalogue():
    old_dir = Path(config.DATA_PROCESSED_DIR)
    new_dir = old_dir.parent / (old_dir.name + "_sharded")
    new_dir.mkdir(parents=True, exist_ok=True)

    data = torch.load(config.EMBEDDINGS_PATH)
    embeddings = data["embeddings"]
    old_filenames = data["filenames"]  # flat filenames, in the exact order embeddings were built

    print(f"Sharding {len(old_filenames)} files from {old_dir} -> {new_dir} "
          f"(shard size = {SHARD_SIZE})...")

    new_filenames = []
    for i, filename in enumerate(old_filenames):
        shard_index = i // SHARD_SIZE
        shard_name = f"shard_{shard_index:02d}"
        shard_dir = new_dir / shard_name
        shard_dir.mkdir(exist_ok=True)

        src = old_dir / filename
        dest = shard_dir / filename
        if not src.exists():
            print(f"WARNING: {src} not found, skipping (embeddings/files may be out of sync)")
            new_filenames.append(None)
            continue

        shutil.copy2(src, dest)
        new_filenames.append(f"{shard_name}/{filename}")

        if (i + 1) % 5000 == 0:
            print(f"  {i + 1}/{len(old_filenames)} copied...")

    # Drop any entries whose source file was missing (keeps embeddings and
    # filenames the same length so indices still match up)
    valid_indices = [i for i, f in enumerate(new_filenames) if f is not None]
    if len(valid_indices) != len(new_filenames):
        print(f"WARNING: dropping {len(new_filenames) - len(valid_indices)} "
              "entries with missing source files.")
        embeddings = embeddings[valid_indices]
        new_filenames = [new_filenames[i] for i in valid_indices]

    new_embeddings_path = str(Path(config.ARTIFACTS_DIR) / "embeddings_sharded.pt")
    torch.save({"embeddings": embeddings, "filenames": new_filenames}, new_embeddings_path)

    n_shards = (len(new_filenames) + SHARD_SIZE - 1) // SHARD_SIZE
    print(f"\nDone. Created {n_shards} shard folders under {new_dir}")
    print(f"New embeddings file (sharded paths) -> {new_embeddings_path}")
    print(f"Example: '{old_filenames[0]}' -> '{new_filenames[0]}'")
    print(
        "\nNext steps:\n"
        f"  1. Confirm {new_dir} looks right (spot-check a shard folder).\n"
        f"  2. Replace the old files:\n"
        f"     - rename {old_dir} to a backup name, or delete it once confident\n"
        f"     - rename {new_dir} to '{old_dir.name}'\n"
        f"     - rename {new_embeddings_path} to '{config.EMBEDDINGS_PATH}' (replacing the old one)\n"
        "  3. Re-run: python src/upload_to_hf.py"
    )


if __name__ == "__main__":
    shard_catalogue()
