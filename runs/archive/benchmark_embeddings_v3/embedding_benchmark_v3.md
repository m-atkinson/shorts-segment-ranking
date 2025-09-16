# Embedding Benchmark v3

Generated: 2025-09-05T16:40:55.024292+00:00 UTC

Dataset: /Users/matthewatkinson/Apps/shorts-model_v3/data/training-data_v3.csv | Samples: 67


| Model | Dim | Device | N | R^2 mean ± std | Spearman mean ± std | Output |
|---|---:|:---:|---:|---:|---:|---|
| sentence-transformers/all-MiniLM-L6-v2 | 384 | mps | 67 | -0.015 ± 0.091 | 0.311 ± 0.230 | shorts_embeddings_all-MiniLM-L6-v2_384_v3.parquet |
| intfloat/e5-base-v2 | 768 | mps | 67 | -0.153 ± 0.064 | -0.088 ± 0.162 | shorts_embeddings_e5-base-v2_768_v3.parquet |
| BAAI/bge-base-en-v1.5 | 768 | mps | 67 | -0.143 ± 0.046 | -0.055 ± 0.215 | shorts_embeddings_bge-base-en-v1.5_768_v3.parquet |

Notes:
- Labels use log1p(view_count).
- Embeddings stored as float32 lists in parquet in this folder.
- CV: Ridge, 5 folds, alpha=1.0.
