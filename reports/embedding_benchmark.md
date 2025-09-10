# Embedding Benchmark

Generated: 2025-09-04T15:05:38.569737+00:00 UTC

Dataset: data/training-data.csv | Samples: 74


| Model | Dim | Device | N | Time (s) | R^2 mean ± std | Spearman mean ± std | Output |
|---|---:|:---:|---:|---:|---:|---:|---|
| sentence-transformers/all-MiniLM-L6-v2 | 384 | mps | 74 | 0.91 | -0.056 ± 0.364 | 0.358 ± 0.263 | data/shorts_embeddings_all-MiniLM-L6-v2_384_v1.parquet |
| intfloat/e5-base-v2 | 768 | mps | 74 | 2.78 | -0.031 ± 0.203 | 0.299 ± 0.260 | data/shorts_embeddings_e5-base-v2_768_v1.parquet |
| BAAI/bge-base-en-v1.5 | 768 | mps | 74 | 2.06 | -0.064 ± 0.255 | 0.310 ± 0.250 | data/shorts_embeddings_bge-base-en-v1.5_768_v1.parquet |

Notes:
- Labels use log1p(view_count).
- Embeddings stored as float32 lists in parquet under data/.
- CV: Ridge, 5 folds, alpha=1.0.
