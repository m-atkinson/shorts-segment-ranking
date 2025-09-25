# Real-Only Holdout Evaluation — v6_holdout_real

Train CSV: data/processed/training-data_v4.3_with-pseudo.csv
Real CSV (pool): data/processed/training-data_v4.1_w-o-outlier.csv
Holdout fraction (of real): 0.2

Train size: 199 (real=54, synth/other=145)
Test size (real only): 13

## Global metrics on holdout (real-only)

- R^2: 0.341
- Spearman: 0.582

## Ranking metrics on holdout (averaged by guest, k=5)

- NDCG@5: 1.000 ± 0.000
- MAP@5: 1.000 ± 0.000
- Recall@5: 1.000 ± 0.000
- Guests evaluated: 2