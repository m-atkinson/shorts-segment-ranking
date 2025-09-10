# Training Report — Guest Feature (v1)

CSV: data/training-data_v5.csv
Model: sentence-transformers/all-MiniLM-L6-v2 | Dim: 384 | Device: mps

## Cross-validated metrics (on y_log)

- R^2 mean ± std: 0.506 ± 0.090
- Spearman mean ± std: 0.763 ± 0.059

## Artifacts

- Regressor: ridge_regressor_guestfeat_v1.pkl
- Guest means embedded in pickle (guest_means, guest_global_mean) for inference