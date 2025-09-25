# Training Report — Guest Feature (v1)

CSV: data/processed/training-data_v5.1_w-o-outlier.csv
Model: sentence-transformers/all-MiniLM-L6-v2 | Dim: 384 | Device: mps

## Cross-validated metrics (on y_log)

- R^2 mean ± std: 0.524 ± 0.200
- Spearman mean ± std: 0.723 ± 0.114

## Artifacts

- Regressor: ridge_regressor_v4.pkl
- Guest means embedded in pickle (guest_means, guest_global_mean) for inference