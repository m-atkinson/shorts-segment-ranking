# Training Report — Guest Feature (v1)

CSV: runs/datasets/training_data_v4_plus_synth_guestfeat_v1.csv
Model: sentence-transformers/all-MiniLM-L6-v2 | Dim: 384 | Device: mps

## Cross-validated metrics (on y_log)

- R^2 mean ± std: 0.507 ± 0.118
- Spearman mean ± std: 0.748 ± 0.031

## Artifacts

- Regressor: ridge_regressor_guestfeat_v1.pkl
- Guest means embedded in pickle (guest_means, guest_global_mean) for inference