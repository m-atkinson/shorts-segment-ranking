# Regression Head Training Report — SynthMix (v1)

Real: runs/datasets/training_data_v4_cleaned.parquet
Synth: runs/synthdata_v1/synth_batch_008_training_format.parquet
Model: sentence-transformers/all-MiniLM-L6-v2 | Dim: 384 | Device: mps
Samples: total=78 real=68 synth=10
Weights: real=1.0 synth=0.5

## Real-only CV metrics (Ridge, 5-fold)

- R^2 mean ± std: -0.015 ± 0.210
- Spearman mean ± std: 0.316 ± 0.153

## Artifact

- Regressor: ridge_regressor_synthmix_v1.pkl