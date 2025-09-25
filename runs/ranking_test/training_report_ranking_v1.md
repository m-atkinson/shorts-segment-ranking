# Training Report — Guest Feature (v1)

CSV: data/processed/training-data_v4.3_with-pseudo.csv
Model: sentence-transformers/all-MiniLM-L6-v2 | Dim: 384 | Device: mps

## Cross-validated metrics (on y_log)

- R^2 mean ± std: 0.767 ± 0.054
- Spearman mean ± std: 0.896 ± 0.026

## Ranking metrics (k=5)

- NDCG@5 mean ± std: 0.991 ± 0.009
- MAP@5 mean ± std: 1.000 ± 0.000
- Recall@5 mean ± std: 0.232 ± 0.005

## Artifacts

- Regressor: ridge_regressor_ranking_v1.pkl
- Guest means embedded in pickle (guest_means, guest_global_mean) for inference