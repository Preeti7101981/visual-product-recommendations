---
title: Visual Product Recommendations
emoji: 👗
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Visual Product Recommendation App

An image-based product recommendation system for a fashion retailer. A user
uploads a photo of a clothing item and receives the top 5 most visually
similar items from the retailer's catalogue, each with a cosine similarity
score.

**Live app:** https://huggingface.co/spaces/Preeti7101981/visual-product-recommendations

## 1. Overview

The system uses a single pretrained ResNet50 convolutional neural network as
a fixed feature extractor (an unsupervised use of a pretrained model: no
labels, no fine-tuning). Every catalogue image is converted into a
2048-dimensional embedding; a user's uploaded image is embedded the same
way, and the catalogue images with the highest cosine similarity to it are
returned as recommendations. The same model, the same embeddings, and the
same full catalogue are used throughout - in the local evaluation pipeline
and in the live deployed app. There is no reduced-catalogue or alternate
model anywhere in this project.

## 2. Dataset and Cleaning Process

The dataset is the Kaggle "Vibrent Clothes Rental" dataset
(https://www.kaggle.com/datasets/kaborg15/vibrent-clothes-rental-dataset).

`src/data_processing.py` processes the raw images:
- **Invalid image removal:** every file is opened and decoded with Pillow;
  files that fail are discarded.
- **Duplicate removal:** a perceptual hash (phash) is computed for each
  valid image; images with a hash already seen are treated as duplicates
  and discarded.

Actual counts from this project's run:

| Stage | Count |
|---|---|
| Raw candidate images found | 50,293 |
| Invalid/corrupted images removed | 0 |
| Duplicate images removed | 19,821 |
| **Final cleaned catalogue** | **31,702** |

Note: duplicate detection is not perfect - see Limitations below, where this
is demonstrated with a concrete example found during evaluation, not just
a theoretical concern.

## 3. Feature Extraction Model

`src/feature_creation.py` defines the single model used everywhere in this
project: **ResNet50** (ImageNet-pretrained weights), with its final
classification layer removed. The 2048-dimensional output of the
penultimate layer (after global average pooling) is used as the image's
embedding. Embeddings are L2-normalized, so cosine similarity between two
embeddings reduces to a simple dot product.

This is an unsupervised use of a supervised-pretrained model: ResNet50's
original ImageNet class labels are never used or referenced anywhere in
this project - only its learned visual feature representation.

## 4. Embedding Generation

`src/build_embeddings.py` applies the model to every image in the cleaned
catalogue and saves the resulting embeddings and their corresponding
filenames to `artifacts/embeddings.pt`. This is computed once; the deployed
app and the evaluation script both load this same file (or the identical
copy hosted on Hugging Face Hub for deployment - see Section 6) rather than
recomputing embeddings on every request.

## 5. Cosine Similarity and Recommendation Process

`src/recommend.py` provides:
- `cosine_similarity_score(a, b)` - the cosine similarity between two
  L2-normalized embeddings (a dot product).
- `recommend_similar_images(...)` - embeds a query image, computes its
  cosine similarity against every catalogue embedding, and returns the
  top-N highest-scoring images. An `exclude_filename` option prevents a
  catalogue image from being recommended to itself when it is also used as
  an evaluation query.

## 6. Evaluation Methodology and Results

`src/evaluate.py` implements the evaluation:
- Randomly selects 3 images from the cleaned catalogue as queries, using
  `config.RANDOM_SEED = 42` (fixed in `config/config.py`, not hardcoded
  elsewhere and not passed via command-line arguments, so the evaluation is
  reproducible by anyone running `run_pipeline.py`).
- For each query, generates the top 5 recommendations, **excluding the
  query image itself** from its own results.
- Saves a plotted figure per query and a combined summary to
  `artifacts/results/recommendation_results.json`.

**Actual results from this run** (target: cosine similarity above 0.7):

| Query | Top 5 cosine similarity scores |
|---|---|
| `97bfc156220748d8b56def5b4b58f743.jpg` | 0.929, 0.922, 0.873, 0.853, 0.853 |
| `0f112da04dc5487abbb4a5776489906c.jpg` | 0.777, 0.769, 0.730, 0.725, 0.722 |
| `0236aadcf2ec45d484970ae923179aa7_409.jpg` | 1.000, 0.877, 0.764, 0.764, 0.762 |

All recommendations scored at or above the 0.7 target. The figures are in
`artifacts/results/`.

The perfect 1.000 score in the third query is notable and discussed in
Limitations below - it is a genuine near-duplicate still present in the
catalogue (a different file than the query image itself; self-exclusion is
confirmed working correctly), not a bug in the evaluation code.

## 7. Deployment Approach

The app (`src/app.py`, a Flask application) is deployed as a **Hugging
Face Space** using the Docker SDK:
https://huggingface.co/spaces/Preeti7101981/visual-product-recommendations

On startup, the app downloads the full cleaned catalogue and its embeddings
from the Hugging Face Hub dataset repo (Section 8) into a local cache, then
serves recommendations from the complete 31,702-image catalogue - the same
catalogue and the same ResNet50 embeddings used in the evaluation above,
not a reduced subset.

A Dockerfile is included for reproducing this deployment. The Space uses
the CPU Basic tier (16GB RAM), which comfortably fits PyTorch, ResNet50,
and the full set of embeddings in memory.

## 8. Dataset Hosting Approach

The full cleaned catalogue (31,702 images, ~4.3GB) and `embeddings.pt`
(~261MB) are too large to practically store in this Git repository, so they
are hosted as a Hugging Face Hub **dataset** repo:
https://huggingface.co/datasets/Preeti7101981/visual-product-recommendations-catalogue

Because Hugging Face Hub repos reject more than 10,000 files in a single
git directory, the images are split into subfolders (`shard_00` -
`shard_03`, ~8,000 files each) via `shard_processed_images.py`. The
`embeddings.pt` file's stored filenames include each image's shard path
(e.g. `shard_00/abc123.jpg`), so the filename-to-embedding-to-image mapping
remains exact and unambiguous regardless of which shard a file sits in.
`src/upload_to_hf.py` performs the upload; `src/app.py` downloads the full
set via `huggingface_hub.snapshot_download` at container startup.

## 9. Limitations and Future Improvements

- **ResNet50 is not fashion-specific.** It was trained for general object
  classification, so it can weight color or background more heavily than
  garment style or cut in some cases.
- **Duplicate detection is imperfect.** Perceptual-hash deduplication missed
  at least one near-duplicate image pair still present in the 31,702-image
  catalogue - demonstrated concretely in the evaluation above (Query 3's
  perfect 1.000 match is a different file than the query, not a
  self-match). A stricter approach (e.g. thresholding on embedding
  similarity directly, rather than perceptual hash alone) would catch these
  more reliably.
- **Catalogue image quality/consistency** (cropping, lighting, background
  clutter) affects embedding quality and therefore match quality.
- **Future improvements:** fine-tune the embedding model on
  fashion-specific data (e.g. `picture_triplets.csv` from the original
  dataset, if it contains labeled similar/dissimilar pairs); try CLIP
  instead of ResNet50, since CLIP is trained with more semantic visual
  supervision; add background removal as a preprocessing step; incorporate
  catalogue metadata (category, color) to re-rank results alongside visual
  similarity.

## 10. Setup and Reproduction Instructions

### Clone and install
```bash
git clone https://github.com/Preeti7101981/visual-product-recommendations.git
cd visual-product-recommendations
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
```
All dependency versions in `requirements.txt` are pinned to versions
confirmed working together.

### Dataset setup
1. Create a free Kaggle account and download the dataset from
   https://www.kaggle.com/datasets/kaborg15/vibrent-clothes-rental-dataset
2. Unzip it and place the contents of its `images` folder into `data/raw/`
   in this project (this folder is created automatically if it doesn't
   exist, but you need to copy the actual image files into it yourself).

### Run the full pipeline
```bash
python src/run_pipeline.py
```
This single command runs, in order: data cleaning -> embedding generation
-> evaluation, creating every output directory it needs automatically. It
reproduces the exact pipeline described in Sections 2-6 above.

### Run the app locally
```bash
python src/app.py
```
Visit http://localhost:5000. On first run this downloads the full
catalogue from Hugging Face Hub (Section 8), which can take a while
depending on your connection.

### Run the tests
```bash
pytest tests/
```

### Redeploy to Hugging Face Spaces (if needed)
See `Dockerfile` and `src/upload_to_hf.py`. In short: push this repository
to a Hugging Face Space created with the Docker SDK; the Space builds the
Dockerfile and runs `src/app.py`, which downloads the catalogue from the
Hugging Face dataset repo at startup.

## Project Structure
```
src/
  data_processing.py     - Data cleaning (Section 2)
  feature_creation.py    - ResNet50 embedding model (Section 3)
  build_embeddings.py    - Embedding generation (Section 4)
  recommend.py           - Cosine similarity + recommendation (Section 5)
  evaluate.py             - Evaluation (Section 6)
  app.py                  - Deployment (Section 7)
  run_pipeline.py         - Runs the full pipeline end to end
  upload_to_hf.py         - Uploads full catalogue to Hugging Face Hub (Section 8)
config/
  config.py               - All paths, model choice, evaluation seed
data/
  raw/                    - Raw Kaggle images (not committed - see Dataset setup)
  processed/              - Cleaned images (not committed, generated by the pipeline)
  test/images/            - Optional manually-supplied test images
artifacts/
  embeddings.pt            - Generated embeddings (not committed, regenerate via the pipeline)
  results/                 - Evaluation figures and recommendation_results.json
tests/
  test_recommend.py        - Unit tests (cosine similarity math, config correctness)
templates/
  index.html                - Web app UI
Dockerfile, .dockerignore   - Hugging Face Space deployment
requirements.txt            - Pinned dependency versions
```
