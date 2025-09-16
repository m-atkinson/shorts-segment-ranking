# Training Report — Guest Feature (v1)

CSV: data/processed/training-data_v4.3_with-pseudo.csv
Model: sentence-transformers/all-MiniLM-L6-v2 | Dim: 384 | Device: mps

## Cross-validated metrics (on y_log)

- R^2 mean ± std: 0.767 ± 0.054
- Spearman mean ± std: 0.896 ± 0.026

## Artifacts

- Regressor: ridge_regressor_v5_top5rand.pkl
- Guest means embedded in pickle (guest_means, guest_global_mean) for inference