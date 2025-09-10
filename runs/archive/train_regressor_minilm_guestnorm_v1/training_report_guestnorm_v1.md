# Training Report — Guest Normalization (v1)

CSV: /Users/matthewatkinson/Apps/shorts-model_v3/data/training-data_v4.xlsx
Model: sentence-transformers/all-MiniLM-L6-v2 | Dim: 384 | Device: mps

## Cross-validated metrics (on residual target)

- R^2 mean ± std: -0.125 ± 0.178
- Spearman mean ± std: 0.256 ± 0.173

## Artifacts

- Regressor: ridge_regressor_guestnorm_v1.pkl
- Guest means embedded in pickle (guest_means, guest_global_mean)