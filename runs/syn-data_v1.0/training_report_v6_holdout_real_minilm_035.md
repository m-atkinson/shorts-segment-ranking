# Real-Only Holdout Evaluation — v6_holdout_real_minilm_035

Train CSV: data/processed/training-data_v4.3_with-pseudo.csv
Real CSV (pool): data/processed/training-data_v4.1_w-o-outlier.csv
Holdout fraction (of real): 0.35
Model: sentence-transformers/all-MiniLM-L6-v2 | Prompt: <none>

Train size: 189 (real=44, synth/other=145)
Test size (real only): 23

## Global metrics on holdout (real-only)

- R^2: 0.654
- Spearman: 0.756

## Ranking metrics on holdout (averaged by guest, k=5)

- NDCG@5: 0.991 ± 0.014
- MAP@5: 0.833 ± 0.204
- Recall@5: 1.000 ± 0.000
- Guests evaluated: 4