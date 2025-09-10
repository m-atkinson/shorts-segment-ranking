# Project documentation for Shorts Model V3

This project finds top-performing YouTube shorts from a long-form transcript using text embeddings and a regression head trained to predict view-related scores.

## Overview

- Inference flow:
  1) Parse transcript into utterances (speaker + timestamps)
  2) Chunk into overlapping segments (~220 tokens, 20% overlap)
  3) Embed each chunk with a sentence-transformer
  4) Score with a trained regression head (Ridge)
  5) Select top-k with diversity (cosine-sim threshold)

- Modularity: modeling is isolated so you can swap out embeddings or the scoring head without touching parsing/chunking.

## Repo layout (key files)

- `shorts_model/`
  - `parsing.py` — parses transcript to utterances from speaker/timestamp markers
  - `chunking.py` — builds overlapping sentence-aware chunks (default 20% overlap)
  - `embedding.py` — wraps sentence-transformers for encoding text
  - `modeling.py` — loads the trained regressor and predicts scores
  - `selection.py` — picks top-k chunks with diversity
- `runs/benchmark_embeddings_v1/` — benchmark script + embeddings + report for model selection
- `runs/train_regressor_minilm_v1/` — training script + embeddings + trained regressor + report
- `runs/infer_minilm_v1/` — inference script + inference outputs (report + JSONL)

## How to run inference

1) Ensure dependencies installed (see `requirements.txt`).
2) Train or place a compatible regressor under `runs/.../*.pkl`.
3) Run the inference script:

```
python runs/infer_minilm_v1/infer_minilm_v1.py \
  --transcript data/transcript_anne-applebaum.txt \
  --top_k 5 \
  --target_tokens 220 \
  --overlap 0.20 \
  --sim_threshold 0.85 \
  --model_name sentence-transformers/all-MiniLM-L6-v2 \
  --regressor_path runs/train_regressor_minilm_v1/ridge_regressor_v1.pkl
```

This writes:
- `runs/infer_minilm_v1/inference_report_v1.md` — human-readable top-k
- `runs/infer_minilm_v1/all_chunks_v1.jsonl` — all chunks with scores

## Swapping in a new modeling component

You can swap either embeddings or the regression head with minimal changes.

### Swap embeddings
- Change the `--model_name` CLI arg in the inference command.
- For training, update the embedding model in the training script (e.g., create `runs/train_regressor_<model>_v1/`), regenerate embeddings, retrain the head, and point `--regressor_path` accordingly.

### Swap regression head
- Implement a new scorer class in `shorts_model/modeling.py` (e.g., ElasticNet or an MLP loaded from a file).
- Save your trained head (e.g., as a pickle with the necessary metadata).
- Update the inference script `--regressor_path` to the new artifact; if needed, extend the script to import the new scorer.

### Recommended file organization pattern
- Keep each experimental run in a `runs/<name>/` folder with:
  - the driver script
  - any artifacts it produces (embeddings, models, markdown reports, JSONL)
  - so it’s obvious which outputs came from which script

## Parser assumptions
- Matches lines like `Speaker Name (MM:SS):` and `(MM:SS):`.
- Removes boilerplate like `PART 1 OF 4 ENDS [...]` and line-number prefixes `123|`.
- If content appears before the first timestamp, its `start_time` is `None`.

## Chunking details
- Sentence-aware splitting with a simple regex (no heavy NLP dependency).
- Target size: ~220 tokens, overlap: 20% (adjustable via CLI).
- Chunk metadata includes `start_time`, `end_time` (approximate from utterances), and speakers.

## Selection and diversity
- Ranks by regressor score.
- Diversity filter: cosine similarity threshold (default 0.85) on embeddings to avoid near-duplicates.

## Training guidance
- Use `runs/benchmark_embeddings_v1/` to compare embedding backbones.
- Use `runs/train_regressor_<model>_v1/` to train ridge/elasticnet/MLP heads; write a training report with cross-validated metrics (Spearman for ranking, R^2 for fit on log scale).
- Start with `log1p(view_count)` as the target. You can experiment later with additional normalization if needed.

## Notes
- All code paths avoid modifying `data/` directly during inference; outputs live next to their scripts under `runs/`.
- The sentence splitter is intentionally simple; you can replace it with spaCy or NLTK if you prefer. If you do, keep the interface the same so chunking remains drop-in.

