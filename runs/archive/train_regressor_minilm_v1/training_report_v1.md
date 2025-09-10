# Regression Head Training Report (v1)

Model: sentence-transformers/all-MiniLM-L6-v2 | Dim: 384 | Device: mps

Samples: 74 | Target: log1p(view_count)


## Cross-validated metrics (Ridge, 5-fold)

- R^2 mean ± std: -0.056 ± 0.364
- Spearman mean ± std: 0.358 ± 0.263

## Artifacts

- Embeddings: shorts_embeddings_all-MiniLM-L6-v2_384_v1.parquet
- Regressor: ridge_regressor_v1.pkl
