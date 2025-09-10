
# Shorts Model V3 — Developer Guide

Welcome! This repo finds top YouTube “shorts” from long-form transcripts using text embeddings and a regression head. This guide helps a new developer get productive quickly.

## High-level architecture

Pipeline steps:
1. Parse: read a long transcript, extract utterances using speaker/timestamp markers.
2. Chunk: split into ~220-token sentence-aware chunks with 20% overlap for context.
3. Embed: compute sentence-transformer embeddings (MiniLM baseline).
4. Score: predict a score with a trained regressor (Ridge). Optionally guest-aware.
5. Select: pick top-k chunks with diversity (cosine-sim threshold) as candidate shorts.

Key modules (shorts_model/):
- parsing.py — transcript → utterances (speaker, start/end times, text)
- chunking.py — utterances → overlapping chunks (target_tokens, overlap)
- embedding.py — wraps sentence-transformers (model_name, batch size)
- modeling.py — scorers:
  - RidgeRegressorScorer (baseline)
  - GuestFeatureScorer (embedding + guest prior feature)
  - GuestNormScorer (residual target + add back guest mean)
  - load_scorer() auto-detects scorer from model pickle metadata
- selection.py — top-k selection with diversity (cosine similarity threshold)

## Repository layout

- shorts_model/ — modular library code
- runs/ — self-contained experiment folders (scripts + artifacts + reports)
  - benchmark_embeddings_v{1,2,3}/ — compare embedding backbones per dataset version
  - train_regressor_minilm_v{1,2,3}/ — baseline training runs (MiniLM + Ridge)
  - infer_minilm_v1/ — inference script + outputs (report + JSONL)
  - dataprep_guest_v1/ — guest-aware data prep
  - train_regressor_minilm_guestnorm_v1/ — guest-normalized residual target training
  - train_regressor_minilm_guestfeat_v1/ — guest feature (target encoding) training
- data/ — input-only datasets and transcripts (do not write outputs here)
- docs/ — additional documentation (see docs/README.md)
- .warp/workflows.yaml — Warp workflows for common tasks

Conventions:
- Never write outputs into data/. All artifacts live under the run folder that created them.
- Every run folder contains: the script, its artifacts (models/embeddings), a markdown report, and (when applicable) a manifest.json.
- Name runs as runs/<stage>_<label>_vN for clear versioning.

## Environment

- Python 3.9+
- Recommended venv: /Users/matthewatkinson/Desktop/032425/venv
- Deps: sentence-transformers, torch, pandas, pyarrow, numpy, scikit-learn, scipy, PyYAML.

Checks:
- `which python` should point to your venv.
- `python -c "import sentence_transformers; print('ok')"`

## Common tasks (Warp)

Workflows are defined in .warp/workflows.yaml so you can run tasks via Warp’s Command Palette.

Examples:
- Data prep (guest): cleans and standardizes guest names, writes a cleaned parquet and a report.
- Train regressor (guest normalization): trains Ridge on residual target (y_log − guest_mean).
- Train regressor (guest feature): trains Ridge on embedding + guest mean feature, target = y_log.
- Train regressor (MiniLM v3 baseline): baseline without guest info.
- Benchmark embeddings (v1/v2/v3): compares MiniLM, e5-base-v2, bge-base-en on a dataset version.
- Infer (baseline or guest-aware): runs full inference with optional --guest.

Each workflow prompts for parameters like dataset path, transcript, regressor path, and guest name.

## Running tasks manually

Baseline inference (MiniLM v3 regressor):
```
python runs/infer_minilm_v1/infer_minilm_v1.py \
  --transcript data/transcript_anne-applebaum.txt \
  --top_k 5 --target_tokens 220 --overlap 0.20 --sim_threshold 0.85 \
  --model_name sentence-transformers/all-MiniLM-L6-v2 \
  --regressor_path runs/train_regressor_minilm_v3/ridge_regressor_v3.pkl
```

Guest-aware inference (guest feature model):
```
python runs/infer_minilm_v1/infer_minilm_v1.py \
  --transcript data/transcript_anne-applebaum.txt \
  --top_k 5 --target_tokens 220 --overlap 0.20 --sim_threshold 0.85 \
  --model_name sentence-transformers/all-MiniLM-L6-v2 \
  --regressor_path runs/train_regressor_minilm_guestfeat_v1/ridge_regressor_guestfeat_v1.pkl \
  --guest "Anne Applebaum"
```

Guest-aware inference (guest normalization model):
```
python runs/infer_minilm_v1/infer_minilm_v1.py \
  --transcript data/transcript_anne-applebaum.txt \
  --top_k 5 --target_tokens 220 --overlap 0.20 --sim_threshold 0.85 \
  --model_name sentence-transformers/all-MiniLM-L6-v2 \
  --regressor_path runs/train_regressor_minilm_guestnorm_v1/ridge_regressor_guestnorm_v1.pkl \
  --guest "Anne Applebaum"
```

Outputs go to:
- `runs/infer_minilm_v1/inference_report_v1.md` (top-k overview)
- `runs/infer_minilm_v1/all_chunks_v1.jsonl` (all chunks with scores)

## Datasets & versions

- training-data.csv — original dataset
- training-data_v2.csv — updated dataset
- training-data_v3.csv — updated dataset
- training-data_v4.xlsx — has guest_name column (Excel)
- transcripts — e.g., data/transcript_anne-applebaum.txt, data/scott_lincicome_294.txt

Guest-aware scripts auto-detect guest column (guest, guest_name, Guest, GuestName). New scripts accept CSV/XLSX/Parquet.

## How to add a new dataset (with guest)

1) Data prep (guest):
```
python runs/dataprep_guest_v1/dataprep_guest_v1.py --csv data/your_file.xlsx
```
2) Train guest-aware variants:
```
python runs/train_regressor_minilm_guestnorm_v1/train_regressor_minilm_guestnorm_v1.py --csv data/your_file.xlsx
python runs/train_regressor_minilm_guestfeat_v1/train_regressor_minilm_guestfeat_v1.py --csv data/your_file.xlsx
```
3) Compare CV Spearman in their reports and choose the winner.
4) Inference: use the selected regressor and pass --guest for guest-aware models.

## Organization & best practices

- One run folder per experiment with scripts, artifacts, report, and (optionally) manifest.json.
- Do not modify previous runs; create a new vN for changes.
- Record dataset path and important hyperparameters in the run’s report/manifest.
- Prefer MiniLM for now; revisit other embeddings after label/head improvements.
- Prefer Spearman in CV for ranking tasks.

## Troubleshooting

- ModuleNotFoundError: sentence_transformers → use your venv python.
- FileNotFoundError for regressor path → point to an existing pickle under runs/.
- ValueError: feature count mismatch (384 vs 385) → using guest-feature regressor without --guest (or old scorer); now supported via --guest and auto-detect.
- Pandas bottleneck warning → non-blocking.

## Extending the project

- New embeddings: add a benchmark run, then a training run; update inference.
- New head (ElasticNet/MLP): add scorer in modeling.py and a training run; save pickles with metadata (target/features) so load_scorer() can auto-detect.
- Label engineering (future): views/day, per-channel normalization, winsorizing outliers, GroupKFold by channel/guest.

## Where to start

- Review run reports under runs/ for current best.
- Use Warp workflows (.warp/workflows.yaml) to prep/train/infer.
- For code changes, start in shorts_model/.
