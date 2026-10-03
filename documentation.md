# Visual Product Recommendation App — Documentation

## Model Summary

This project implements an image-based product recommendation system for a
fashion retailer, using a single consistent approach throughout: a
pretrained ResNet50 convolutional network (ImageNet weights, classification
layer removed) as a fixed feature extractor, producing L2-normalized
2048-dimensional embeddings compared via cosine similarity. The raw
catalogue contained 50,293 candidate images; after removing 0 invalid and
19,821 duplicate images (identified via perceptual hashing), the final
cleaned catalogue contains 31,702 unique images. Embeddings for the full
catalogue were computed once and are used identically in evaluation and in
the live deployed application - there is no reduced catalogue or
alternate model anywhere in this system. Evaluation was performed on 3
randomly selected catalogue images (seed fixed in `config/config.py` for
reproducibility), each excluded from its own recommendations. All 15
resulting recommendations (5 per query, 3 queries) scored at or above the
0.7 cosine similarity target, with scores ranging from 0.722 to a perfect
1.000. The live application, hosted as a Hugging Face Space, downloads
this same full 31,702-image catalogue and the same embeddings from a
Hugging Face Hub dataset repository at startup and serves recommendations
from it directly.

**[Insert one or more of the figures from `artifacts/results/` here - each
shows a query image alongside its top-5 recommendations and their cosine
similarity scores.]**

## Limitations and Future Improvements

ResNet50 was trained for general object classification rather than
fashion-specific similarity, so it can weight color or background more
heavily than garment style or cut in some cases. More concretely, the
perfect 1.000 score obtained for one evaluation query is not a self-match
(the query's own filename is correctly excluded from its results) but a
genuine near-duplicate image still present in the cleaned catalogue -
direct evidence that perceptual-hash deduplication, while effective
(removing 19,821 of 50,293 raw images), does not catch every near-duplicate
pair. A stricter approach, such as thresholding directly on embedding
similarity rather than relying on perceptual hashing alone, would likely
catch these remaining cases. Catalogue image quality and consistency
(cropping, lighting, background clutter) also affects embedding quality
and therefore recommendation quality. Future improvements could include
fine-tuning the embedding model on fashion-specific labeled data (e.g. the
dataset's own `picture_triplets.csv`, if suitable), replacing ResNet50 with
CLIP (trained with more semantic visual supervision and likely to group
images by style more naturally), adding background removal as a
preprocessing step, and incorporating catalogue metadata (category, color)
to re-rank visually similar results.
