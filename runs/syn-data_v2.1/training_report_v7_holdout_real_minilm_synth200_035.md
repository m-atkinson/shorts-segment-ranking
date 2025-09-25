# Real-Only Holdout Evaluation — v7_holdout_real_minilm_synth200_035

Train CSV: data/processed/training-data_v7_abs200.csv
Real CSV (pool): data/processed/training-data_v5.1_w-o-outlier.csv
Holdout fraction (of real): 0.35
Model: sentence-transformers/all-MiniLM-L6-v2 | Prompt: <none>

Train size: 251 (real=51, synth/other=200)
Test size (real only): 26

## Global metrics on holdout (real-only)

- R^2: 0.664
- Spearman: 0.874

## Ranking metrics on holdout (averaged by guest, k=5)

- NDCG@5: 0.999 ± 0.003
- MAP@5: 0.972 ± 0.068
- Recall@5: 1.000 ± 0.000
- Guests evaluated: 7