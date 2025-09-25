# Real-Only Holdout Evaluation — v8_holdout_real_minilm_synth1000_035

Train CSV: data/processed/training-data_v8_abs1000.csv
Real CSV (pool): data/processed/training-data_v5.1_w-o-outlier.csv
Holdout fraction (of real): 0.35
Model: sentence-transformers/all-MiniLM-L6-v2 | Prompt: <none>

Train size: 1051 (real=51, synth/other=1000)
Test size (real only): 26

## Global metrics on holdout (real-only)

- R^2: 0.650
- Spearman: 0.878

## Ranking metrics on holdout (averaged by guest, k=5)

- NDCG@5: 0.997 ± 0.007
- MAP@5: 0.988 ± 0.029
- Recall@5: 1.000 ± 0.000
- Guests evaluated: 7