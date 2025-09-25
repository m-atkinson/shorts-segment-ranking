# Real-Only Holdout Evaluation — v6_holdout_real_minilm_mix40_035

Train CSV: data/processed/training-data_mix_40.csv
Real CSV (pool): data/processed/training-data_v5.1_w-o-outlier.csv
Holdout fraction (of real): 0.35
Model: sentence-transformers/all-MiniLM-L6-v2 | Prompt: <none>

Train size: 82 (real=51, synth/other=31)
Test size (real only): 26

## Global metrics on holdout (real-only)

- R^2: 0.630
- Spearman: 0.798

## Ranking metrics on holdout (averaged by guest, k=5)

- NDCG@5: 0.991 ± 0.015
- MAP@5: 0.882 ± 0.177
- Recall@5: 1.000 ± 0.000
- Guests evaluated: 7