# Real-Only Holdout Evaluation — v6_holdout_real_minilm_mix20_035

Train CSV: data/processed/training-data_mix_20.csv
Real CSV (pool): data/processed/training-data_v5.1_w-o-outlier.csv
Holdout fraction (of real): 0.35
Model: sentence-transformers/all-MiniLM-L6-v2 | Prompt: <none>

Train size: 66 (real=51, synth/other=15)
Test size (real only): 26

## Global metrics on holdout (real-only)

- R^2: 0.568
- Spearman: 0.765

## Ranking metrics on holdout (averaged by guest, k=5)

- NDCG@5: 0.988 ± 0.020
- MAP@5: 0.842 ± 0.201
- Recall@5: 1.000 ± 0.000
- Guests evaluated: 7