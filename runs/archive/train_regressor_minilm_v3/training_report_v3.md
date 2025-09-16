# Regression Head Training Report (v3)

Model: sentence-transformers/all-MiniLM-L6-v2 | Dim: 384 | Device: mps

Samples: 67 | Target: log1p(view_count)


## Cross-validated metrics (Ridge, 5-fold)

- R^2 mean ± std: -0.015 ± 0.091
- Spearman mean ± std: 0.311 ± 0.230

## Artifacts

- Embeddings: shorts_embeddings_all-MiniLM-L6-v2_384_v3.parquet
- Regressor: ridge_regressor_v3.pkl
