# Regression Head Training Report (v2)

Model: sentence-transformers/all-MiniLM-L6-v2 | Dim: 384 | Device: mps

Samples: 73 | Target: log1p(view_count)


## Cross-validated metrics (Ridge, 5-fold)

- R^2 mean ± std: -0.027 ± 0.111
- Spearman mean ± std: 0.305 ± 0.120

## Artifacts

- Embeddings: shorts_embeddings_all-MiniLM-L6-v2_384_v2.parquet
- Regressor: ridge_regressor_v2.pkl
