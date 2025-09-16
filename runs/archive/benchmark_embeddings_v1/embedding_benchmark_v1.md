# Embedding Benchmark v1

Generated: 2025-09-04T16:14:28.676414+00:00 UTC

Dataset: /Users/matthewatkinson/Apps/shorts-model_v3/data/training-data.csv | Samples: 74


| Model | Dim | Device | N | R^2 mean ± std | Spearman mean ± std | Output |
|---|---:|:---:|---:|---:|---:|---|
| sentence-transformers/all-MiniLM-L6-v2 | 384 | mps | 74 | -0.056 ± 0.364 | 0.358 ± 0.263 | shorts_embeddings_all-MiniLM-L6-v2_384_v1.parquet |
| intfloat/e5-base-v2 | 768 | mps | 74 | -0.031 ± 0.203 | 0.299 ± 0.260 | shorts_embeddings_e5-base-v2_768_v1.parquet |
| BAAI/bge-base-en-v1.5 | 768 | mps | 74 | -0.064 ± 0.255 | 0.310 ± 0.250 | shorts_embeddings_bge-base-en-v1.5_768_v1.parquet |

Notes:
- Labels use log1p(view_count).
- Embeddings stored as float32 lists in parquet in this folder.
- CV: Ridge, 5 folds, alpha=1.0.
