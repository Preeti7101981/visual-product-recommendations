"""
Data Processing and Cleaning
=============================
Reads raw clothing images, removes invalid (corrupted/unreadable) images and
duplicate images (via perceptual hashing), and writes the cleaned set to
config.DATA_PROCESSED_DIR along with a manifest CSV.

Can be run standalone:
    python src/data_processing.py
or as part of the full pipeline:
    python src/run_pipeline.py
"""

import sys
from pathlib import Path

import imagehash
import pandas as pd
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import config

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def is_valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            img.convert("RGB")
        return True
    except (UnidentifiedImageError, OSError, ValueError):
        return False


def compute_phash(path: Path):
    with Image.open(path) as img:
        return imagehash.phash(img.convert("RGB"))


def process_images(raw_dir: str = None, out_dir: str = None) -> pd.DataFrame:
    raw_dir = Path(raw_dir or config.DATA_RAW_DIR)
    out_dir = Path(out_dir or config.DATA_PROCESSED_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    candidate_files = [
        p for p in raw_dir.rglob("*")
        if p.suffix.lower() in VALID_EXTENSIONS and p.is_file()
    ]
    print(f"Found {len(candidate_files)} candidate image files in {raw_dir}.")

    if not candidate_files:
        print(
            f"WARNING: no images found in {raw_dir}. "
            "Download the Kaggle dataset and place the 'images' folder contents there "
            "(see README.md 'Dataset setup')."
        )
        return pd.DataFrame(columns=["filename", "source_path", "phash"])

    seen_hashes = set()
    records = []
    n_invalid = 0
    n_duplicate = 0

    for path in tqdm(candidate_files, desc="Processing images", unit="img"):
        if not is_valid_image(path):
            n_invalid += 1
            continue
        try:
            phash = compute_phash(path)
        except Exception:
            n_invalid += 1
            continue
        if phash in seen_hashes:
            n_duplicate += 1
            continue
        seen_hashes.add(phash)

        dest = out_dir / path.name
        if dest.exists():
            dest = out_dir / f"{path.stem}_{len(records)}{path.suffix}"
        import shutil
        shutil.copy2(path, dest)
        records.append({"filename": dest.name, "source_path": str(path), "phash": str(phash)})

    manifest = pd.DataFrame(records)
    manifest.to_csv(config.CLEANED_MANIFEST_CSV, index=False)

    print(f"Removed {n_invalid} invalid images and {n_duplicate} duplicate images.")
    print(f"Kept {len(manifest)} clean images -> {out_dir}")
    print(f"Manifest written to {config.CLEANED_MANIFEST_CSV}")
    return manifest


def main():
    config.ensure_required_dirs()
    process_images()


if __name__ == "__main__":
    main()
