# One-Time Historical Utilities

These scripts are **not** part of the main pipeline (`python src/run_pipeline.py`)
and a fresh clone of this project does not need to run any of them. They
were used once, during development, to fix specific historical issues:

- `shard_processed_images.py` - retroactively sharded an existing flat
  catalogue into subfolders. No longer needed going forward:
  `src/data_processing.py` now shards images into `shard_NN` subfolders
  automatically as part of normal cleaning, so a fresh run never produces
  a flat catalogue in the first place.
- `migrate_old_embeddings.py` - converted an embeddings file saved with
  full file paths (an earlier, since-replaced format) into the current
  relative-path format, without recomputing embeddings.
- `cleanup_hf_stray_files.py` - removed stray files left on the Hugging
  Face Hub dataset repo after an initial upload attempt partially failed
  (before `data_processing.py` sharded automatically).

Kept for transparency/reference rather than deleted outright.
