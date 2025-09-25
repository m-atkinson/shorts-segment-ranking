# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Repository Overview

This is a YouTube Shorts prediction model that analyzes long-form podcast transcripts to identify the 5 most promising segments for creating viral YouTube Shorts. The system combines semantic text embeddings with machine learning to predict view counts and recommend diverse, high-potential content segments.

**Current Performance**: Model v5 achieves R² = 0.767 and Spearman correlation = 0.896 on real YouTube Shorts performance data.

## Core Architecture

The system follows a modular pipeline architecture:

```
Raw Transcript (.txt/.yaml)
        ↓
Transcript Parsing (speaker/timestamp detection)
        ↓
Chunking (220 tokens, 20% overlap, sentence-aware)
        ↓
Text Embedding (sentence-transformers/all-MiniLM-L6-v2, 384-dim)
        ↓
Guest Target Encoding (historical performance baseline)
        ↓
Ridge Regression Prediction (log view count)
        ↓
Diversity-Aware Selection (top-5, cosine similarity filtering)
```

Key insight: **Guest identity is the strongest predictor** (coefficient=0.948), setting baseline expectations, while text embeddings determine relative ranking within transcripts.

## Project Structure

- **`shorts_model/`** - Core processing modules:
  - `parsing.py` - Transcript parsing with speaker/timestamp detection
  - `parsing_yaml.py` - Advanced YAML transcript parsing with guest extraction
  - `chunking.py` - Sentence-aware chunking with overlap
  - `embedding.py` - Sentence transformer embedding wrapper
  - `modeling.py` - Multiple scorer types (Ridge, GuestFeature, GuestNorm)
  - `selection.py` - Diversity-aware top-k selection
- **`shorts_model/inference/`** - Inference scripts
- **`shorts_model/modeling/`** - Training scripts
- **`configs/`** - Configuration files and example commands
- **`reports/`** - Performance benchmarks and analysis
- **`transcript-pipeline_full-videos/`** - Data collection utilities

## Development Commands

### Environment Setup
```bash
pip install -r requirements.txt
```

### Training a New Model
```bash
# Train guest-aware model (recommended)
python3 shorts_model/modeling/train.py \
    --csv data/processed/training-data_v4.3_with-pseudo.csv \
    --outdir runs/my_experiment \
    --name my_model_v1
```

### Running Inference
```bash
# Current best model (v5)
python3 shorts_model/inference/infer_minilm_v1.py \
    --transcript data/raw/transcript_example.txt \
    --regressor_path runs/archive/v5/ridge_regressor_v5_top5rand.pkl \
    --guest "Anne Applebaum" \
    --top_k 5
```

### Running Tests
```bash
# No formal test suite - validation via cross-validation in training scripts
python3 shorts_model/modeling/train.py --csv <test_data.csv>
```

### Training with Ranking Evaluation
```bash
# Standard training now includes NDCG@5, MAP@5, Recall@5 metrics
python3 shorts_model/modeling/train.py \
    --csv data/processed/training-data_v4.3_with-pseudo.csv \
    --outdir runs/ranking_experiment \
    --name model_with_ranking
```

### Benchmarking Embeddings
```bash
# Compare different embedding models
python3 configs/benchmark_embeddings.py --csv data/training-data.csv
```

## Key Technical Details

### Model Architecture Specifics
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)
- **ML Algorithm**: Ridge Regression (α=1.0) for robustness and interpretability
- **Target Variable**: Log-transformed view counts to handle skewed distribution
- **Feature Engineering**: 384-dim text embeddings + 1-dim guest target encoding
- **Validation**: 5-fold cross-validation with leakage-safe target encoding

### Chunking Strategy
- **Target Size**: ~220 tokens (optimized for 60-second video segments)
- **Overlap**: 20% to capture boundary information
- **Sentence-Aware**: Splits on sentence boundaries using regex
- **Metadata**: Preserves timestamps, speaker labels, and token counts

### Selection Algorithm
- **Primary Ranking**: Ridge regression scores (log view count predictions)
- **Diversity Filter**: Cosine similarity threshold = 0.85 to avoid redundant selections
- **Guest Awareness**: Models incorporate historical guest performance as baseline

### Model Variants Available
1. **RidgeRegressorScorer** - Embeddings only (baseline)
2. **GuestFeatureScorer** - Embeddings + guest target encoding (recommended)
3. **GuestNormScorer** - Residual target modeling

The system automatically detects model type from pickle metadata via `load_scorer()`.

### Cross-Validation Strategy
```python
# Leakage-safe target encoding prevents data leakage
for train_idx, test_idx in kfold.split(X):
    train_guest_means = compute_guest_means(y[train_idx], guests[train_idx])
    # Apply encoding consistently to train/test splits
```

### Performance Characteristics
- **Processing Speed**: ~50 chunks per second on M1 MacBook Pro
- **Memory Requirements**: ~1-2GB RAM for inference, ~2GB for training
- **Hardware**: CPU sufficient, MPS/GPU optional for embeddings

### Data Format Requirements
- **Training CSV**: Must include `video_id`, `view_count`, `guest_name`, `transcription` columns
- **Transcript Formats**: Supports both plain text and structured YAML
- **Speaker Detection**: Regex patterns for "Speaker Name (MM:SS):" format

## Synthetic Data Generation

Updated approach focused on high‑quality, model‑relevant positives, with clear provenance and auditability.

- Purpose: Augment the small real dataset with additional, high‑signal training rows derived from long‑form transcripts.
- Scoring model: Use the current best ridge regressor (guest‑aware) to score chunks; optionally run a second scorer (e.g., EmbeddingGemma+prompt) and combine scores (mean or max) to reduce single‑model bias.
- Per‑transcript selection: Rank all chunks for a transcript and take the top‑5 candidates (post diversity filter), then randomly select the final keep set from those top candidates. This preserves quality while avoiding over‑fitting to a single deterministic pick.
- Diversity: Apply cosine‑similarity filtering when building the per‑transcript top‑5 (e.g., sim ≤ 0.85) to avoid near‑duplicates.
- Validation (strict, YAML): Verify that the emitted chunk text is present in the source transcript (whitespace‑insensitive match after normalization). Rows failing validation are dropped.
- Guest attribution: Infer/confirm guest per transcript (from YAML metadata or speaker heuristic) and pass guest to the scorer so guest‑aware features are used consistently.
- Output format: Emit training‑ready CSV rows with columns: `video_id`, `view_count` (pred from log via expm1), `guest_name`, `transcription`.
- Provenance: Store source file name, chunk_id, model_name, and timestamp in an internal manifest for traceability (optional alongside the CSV).

Notes and options:
- Committee scoring (optional): Average or max across multiple scorers (e.g., MiniLM and Gemma) before ranking to reduce model‑specific artifacts.
- Balancing (optional): If desired, sample uniformly across transcripts/guests when choosing from the per‑transcript top‑5 to avoid over‑representing frequent guests.
- Ratio policy: The synthetic/real ratio is a tuning knob, not fixed in this doc. Use ablations on a frozen real‑only holdout to choose the ratio that improves ranking metrics (MAP@5) without hurting robustness.

## Guest Performance Insights

Top performing guests (by average views):
- **Anne Applebaum**: 18,960 avg views (authoritarianism expert)
- **John Bolton**: 13,554 avg views (foreign policy)
- **James Carville**: 11,011 avg views (Democratic politics)

## Evaluation Philosophy

### Current Ranking Evaluation System
The standard training pipeline now includes ranking evaluation alongside traditional regression metrics:

**Ranking Metrics:**
- **NDCG@5**: Normalized Discounted Cumulative Gain at rank 5
- **MAP@5**: Mean Average Precision at rank 5  
- **Recall@5**: Recall at rank 5

**Current Implementation:**
- **Standard K-Fold**: 5-fold cross-validation with ranking metrics calculated per fold
- Integrated into the main `train.py` script for simplicity
- Reports both regression (R², Spearman) and ranking (NDCG@5, MAP@5, Recall@5) metrics
- Results saved in both Markdown reports and JSON manifests

**Future Enhancements:**
- Leave-Guest-Out and Leave-Episode-Out validation strategies
- Time-based splits for temporal evaluation
- Advanced ranking algorithms (LambdaMART, etc.)

## Hardware and Dependencies

**Key Dependencies:**
- `sentence-transformers>=3.0.0` - Text embeddings
- `scikit-learn>=1.4.0` - Ridge regression and cross-validation  
- `torch>=2.2.0` - PyTorch backend for transformers
- `pandas>=2.2.0` - Data manipulation
- `PyYAML>=6.0.0` - YAML transcript parsing

**Device Support:**
- CPU: Full functionality
- Apple Silicon (MPS): Automatic detection and usage
- CUDA: Supported via PyTorch

The system is designed for research and production use in content recommendation pipelines, with emphasis on interpretability, robustness, and preventing overfitting in small-data regimes.
