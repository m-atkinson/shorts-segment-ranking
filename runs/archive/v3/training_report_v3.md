# Training Report — Guest Feature (v1)

CSV: data/processed/training-data_v4.1_w-o-outlier.csv
Model: sentence-transformers/all-MiniLM-L6-v2 | Dim: 384 | Device: mps

## Cross-validated metrics (on y_log)

- R^2 mean ± std: 0.177 ± 0.265
- Spearman mean ± std: 0.592 ± 0.102

## Artifacts

- Regressor: ridge_regressor_v3.pkl
- Guest means embedded in pickle (guest_means, guest_global_mean) for inference