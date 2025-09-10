# Embedding Benchmark v2

Generated: 2025-09-05T15:56:28.870483+00:00 UTC

Dataset: /Users/matthewatkinson/Apps/shorts-model_v3/data/training-data_v2.csv | Samples: 73


| Model | Dim | Device | N | R^2 mean ± std | Spearman mean ± std | Output |
|---|---:|:---:|---:|---:|---:|---|
| sentence-transformers/all-MiniLM-L6-v2 | 384 | mps | 73 | -0.027 ± 0.111 | 0.305 ± 0.120 | shorts_embeddings_all-MiniLM-L6-v2_384_v2.parquet |
| intfloat/e5-base-v2 | 768 | mps | 73 | -0.105 ± 0.149 | 0.152 ± 0.234 | shorts_embeddings_e5-base-v2_768_v2.parquet |
| BAAI/bge-base-en-v1.5 | 768 | mps | 73 | -0.091 ± 0.145 | 0.162 ± 0.206 | shorts_embeddings_bge-base-en-v1.5_768_v2.parquet |

Notes:
- Labels use log1p(view_count).
- Embeddings stored as float32 lists in parquet in this folder.
- CV: Ridge, 5 folds, alpha=1.0.
