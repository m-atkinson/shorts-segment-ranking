

python3 shorts_model/features/synthdata_v1/generate_synth_batch_v1.py \
  --yaml_dir data/raw/transcripts \
  --regressor_path runs/train_regressor_minilm_guestfeat_v1/ridge_regressor_guestfeat_v1.pkl \
  --model_name sentence-transformers/all-MiniLM-L6-v2 \
  --output_stem data/interim/synth_batch_002



# Directory Structure 
https://cookiecutter-data-science.drivendata.org/?utm_source=chatgpt.com
This is not exactly how it is setup, but it is close

├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         {{ cookiecutter.module_name }} and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── {{ cookiecutter.module_name }}   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes {{ cookiecutter.module_name }} a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations

---

# YouTube Shorts Prediction Model
A ranking assistant that proposes diverse, high-potential segments from long-form podcast transcripts.
 

## Goal

This project addresses the challenge of scaling short-form content creation from long-form podcasts, specifically the time an editor spends searching for segments to create shorts. 

**Key Achievement**: The current model (v5) achieves **R² = 0.767** and **Spearman correlation = 0.896** on actual YouTube Shorts performance data.

## Problem Statement

- **Input**: Hour-long podcast transcripts (e.g., "Conversations with Bill Kristol")
- **Challenge**: Identify 5 segments most likely to generate high YouTube Shorts engagement
- **Output**: Ranked list of ~220-word segments with predicted view counts and timestamps

## Technical Approach

### Architecture

```
Podcast Transcript (.txt/.yaml)
        ↓
Transcript Parsing & Speaker Detection
        ↓
Chunking (~220 words, 20% overlap)
        ↓
Sentence Transformers Embedding (384-dim)
        ↓
Guest Target Encoding (1-dim)
        ↓
Ridge Regression → log(view_count)
        ↓
Diversity-Aware Selection (top-5)
```

### Core Components

1. **Text Processing Pipeline**
   - Parses timestamps and speaker labels from raw transcripts
   - Creates overlapping chunks optimized for ~60-second segments
   - Handles both plain text and structured YAML formats

2. **Feature Engineering**
   - **Semantic Embeddings**: Uses `sentence-transformers/all-MiniLM-L6-v2` for 384-dimensional text representations
   - **Guest Features**: Target encoding based on historical guest performance
   - **Combined Features**: 385 total dimensions (384 + 1)

3. **Machine Learning Model**
   - **Algorithm**: Ridge Regression (α=1.0) for robustness and interpretability
   - **Target**: Log-transformed view counts to handle skewed distribution
   - **Validation**: 5-fold cross-validation for reliable performance estimates

4. **Selection Strategy**
   - Diversity-aware ranking using cosine similarity (threshold=0.85)
   - Prevents selection of multiple similar segments
   - Balances high scores with content variety

## Performance Metrics

### Model Performance (v5)
- **R² Score**: 0.767 ± 0.054 (explains 76.7% of variance)
- **Spearman Correlation**: 0.896 ± 0.026 (excellent ranking quality)
- **Training Data**: 212 actual YouTube Shorts with view counts (881-24,847 views)
- **Cross-Validation**: 5-fold with leakage-safe target encoding

### Feature Importance
- **Guest Identity**: Strongest predictor across different guests (coefficient = 0.948)
- **Text Embeddings**: Determine ranking within individual transcripts during inference
- **Combined Effect**: Guest sets baseline expectation, text content drives segment selection
- **Top Performing Guests**: Anne Applebaum (18,960 avg views), John Bolton (13,554), James Carville (11,011)

## Dataset

### Training Data Structure
```csv
video_id,view_count,guest_name,transcription
kqH4oM8uhqI,2025,Ryan Goodman,"We're seeing things that I would never have imagined..."
LheEa0iNiI0,1232,Ryan Goodman,"the unitary executive, the idea that the president..."
```

### Content Characteristics
- **Domain**: Political commentary and analysis
- **Length**: ~197 words average (190-300 word range)
- **Guests**: 30+ regular contributors (academics, former officials, journalists)
- **Topics**: Ukraine, authoritarianism, US politics, foreign policy

## Installation & Usage

### Prerequisites
```bash
pip install -r requirements.txt
```

### Key Dependencies
- `sentence-transformers>=3.0.0` - Text embeddings
- `scikit-learn>=1.4.0` - Ridge regression and cross-validation
- `torch>=2.2.0` - PyTorch backend for transformers
- `pandas>=2.2.0` - Data manipulation
- `numpy>=1.26.0` - Numerical computations

### Training a New Model
```bash
python3 shorts_model/modeling/train.py \
    --csv data/processed/training-data_v4.3_with-pseudo.csv \
    --outdir runs/my_experiment \
    --name my_model_v1
```

### Running Inference
```bash
python3 shorts_model/inference/infer_minilm_v1.py \
    --transcript data/raw/transcript_example.txt \
    --regressor_path runs/v5/ridge_regressor_v5_top5rand.pkl \
    --guest "Anne Applebaum" \
    --top_k 5
```

### Generating Synthetic Training Data
```bash
python3 shorts_model/features/synthdata_v1/generate_synth_batch_v1.py \
    --yaml_dir data/raw/transcripts \
    --regressor_path runs/v5/ridge_regressor_v5_top5rand.pkl \
    --model_name sentence-transformers/all-MiniLM-L6-v2 \
    --output_stem data/interim/synth_batch_new
```

## Example Output

### Inference Report
```markdown
# Top 5 Recommended Segments

## #1 | Score: 10.161 | Duration: 8:32-9:15 | Tokens: 223
**Speakers**: Anne Applebaum

"So for them, an occupation is the imposition of a totalitarian regime, 
and they can't live there, it will be the end. So they will keep fighting. 
And for Russia, it remains a war of choice..."

## #2 | Score: 10.103 | Duration: 15:22-16:08 | Tokens: 274
**Speakers**: Anne Applebaum, Bill Kristol

"I mean, we have a lot of leverage on them, but they have no one to negotiate with. 
I don't think the Russians are negotiating about anything until they have concluded 
that they can't win..."
```

## Model Evolution

### Version History
- **v1**: Baseline MiniLM + simple regression (R² ~0.3)
- **v2**: Added guest name as categorical feature (R² ~0.4)
- **v3**: Guest target encoding with cross-validation (R² ~0.5)
- **v4**: Synthetic data augmentation (R² 0.524 ± 0.200)
- **v5**: Improved synthetic data quality (R² 0.767 ± 0.054) ✅ **Current**

### Key Innovations
1. **Guest-Aware Modeling**: Recognition that guest identity is the strongest predictor
2. **Target Encoding with CV**: Leakage-safe encoding of categorical guest features
3. **Synthetic Data Generation**: Using trained models to expand training set
4. **Diversity-Aware Selection**: Preventing redundant recommendations

## Synthetic Data Generation: Strengths and Limitations

### **The Approach**
The system generates additional training data by:
1. **Self-Bootstrap**: Using trained models to score unprocessed transcript chunks
2. **Quality Filtering**: Selecting only high-scoring segments (top 25th percentile)
3. **Diversity Control**: Maximum 2 segments per transcript, cosine similarity deduplication (≥0.95)
4. **Guest Assignment**: Assigning predicted view counts based on guest identity

### **Example Synthetic Generation Process**
```bash
# Input: 2,719 candidate chunks from unlabeled transcripts
# Filter: Score threshold p75 = 8.753 (log view count)
# Output: 10 high-quality synthetic examples
# Rejection: 126 due to per-transcript limits, 0 due to low quality
```

### **Synthetic Data Characteristics**
- **Quality Range**: 13,551-29,461 predicted views (high-performing segments only)
- **Guest Focus**: Primarily Anne Applebaum (6), John Bolton (3), James Carville (1)
- **Content Type**: Complex geopolitical analysis, similar to highest-performing real data
- **Volume**: ~10-50 additional examples per batch (vs. 212 real examples)

### **Strengths** ✅

1. **Data Augmentation in Small Dataset Regime**
   - Original dataset: Only 212 labeled examples across 30+ guests
   - Some guests have <5 examples → synthetic data helps with rare guest coverage
   - Addresses class imbalance (few high-performing examples)

2. **Quality Assurance via Self-Selection**
   - Only selects segments the model already predicts as high-quality
   - Filters out ~99.6% of candidate segments (10 selected from 2,719)
   - Deduplication prevents repetitive content

3. **Domain Consistency**
   - Generated from same podcast format and guest pool
   - Maintains realistic content length and style
   - Preserves conversational structure and political commentary tone

4. **Performance Validation**
   - v4 → v5 improvement (R² 0.524 → 0.767) coincided with synthetic data integration
   - Cross-validation prevents direct overfitting (synthetic data not in all folds)

### **Limitations** ⚠️

1. **Circular Reasoning Risk**
   - **Problem**: Model generates its own training data based on its predictions
   - **Consequence**: May amplify existing biases rather than learning new patterns
   - **Example**: If model incorrectly weights certain phrases, synthetic data reinforces this

2. **Distribution Shift**
   - **Synthetic view counts**: 13k-29k (artificially high)
   - **Real view counts**: 881-24,847 (natural distribution)
   - **Risk**: Model may become overconfident about what constitutes "high performance"

3. **Limited Diversity**
   - **Guest Concentration**: 60% Anne Applebaum synthetic data vs. 9% real data proportion
   - **Content Similarity**: All synthetic examples scored highly by same model
   - **Missing Edge Cases**: No synthetic examples of surprising low/high performers

4. **Evaluation Challenges**
   - **Impossible to validate**: Synthetic segments never actually became YouTube Shorts
   - **Assumption**: High model scores → high real-world performance
   - **Risk**: Model could be confident but wrong about synthetic examples

### **Mitigation Strategies**

1. **Conservative Mixing Ratios**
   - Keep synthetic data <20% of total training set
   - Maintain strong representation of real, measured performance data

2. **Cross-Validation Isolation**
   - Ensure synthetic data doesn't leak across CV folds
   - Test model performance on purely real-world held-out data

3. **Regular Recalibration**
   - Periodically retrain base models on real data only
   - Generate fresh synthetic batches to prevent staleness

4. **Performance Monitoring**
   - Track if synthetic data improves or hurts performance on real test sets
   - Monitor for overconfidence in model predictions

### **Alternative Approaches Considered**

- **Manual Curation**: Too labor-intensive for content experts
- **Random Sampling**: Would include many low-quality segments
- **External Data Sources**: Different podcast formats might not transfer
- **Data Augmentation**: Text perturbations could change semantic meaning

**Conclusion**: Synthetic data generation is a powerful but potentially dangerous technique. In this low-data regime (212 examples), it provides valuable augmentation, but requires careful monitoring to prevent the model from "drinking its own Kool-Aid."

## Content Analysis Insights

### High-Performing Content Patterns
- **Controversial Statements**: "Putin is still a master manipulator of Trump" (16,880 views)
- **Counter-Narrative Insights**: "Europe now gives more aid to Ukraine than the US" (16,199 views)
- **Expert Analysis**: Direct responses to probing questions from Bill Kristol
- **Current Events**: References to breaking news, recent developments

### Guest Performance Analysis
| Guest | Avg Views | Episodes | Expertise Area |
|-------|-----------|----------|----------------|
| Anne Applebaum | 18,960 | 20 | Authoritarianism, Eastern Europe |
| John Bolton | 13,554 | 15 | Foreign Policy, Trump Administration |
| James Carville | 11,011 | 22 | Democratic Politics, Strategy |
| Aaron Friedberg | 6,789 | 15 | China, Geopolitics |

## Technical Details

### Model Architecture Specifics
- **Ridge Alpha**: 1.0 (optimal via cross-validation)
- **Embedding Model**: all-MiniLM-L6-v2 (384 dimensions, English-optimized)
- **Chunking**: 220 target tokens, 20% overlap, sentence-boundary aware
- **Guest Encoding**: Historical mean log(view_count) with global fallback
- **Selection Threshold**: 0.85 cosine similarity for diversity filtering

### Feature Roles: Training vs Inference

**During Training (cross-guest prediction):**
- Guest feature (coefficient=0.948) explains most variance between different guests
- Text embeddings (384 coefficients) capture content patterns within and across guests

**During Inference (single transcript ranking):**
- Guest feature is constant → provides baseline score for all segments
- Text embeddings do all the ranking work → determine which segments score highest
- Example: All Anne Applebaum segments get +0.948×(her_guest_mean), then text content determines relative ordering

### Cross-Validation Strategy
```python
# Leakage-safe target encoding
for train_idx, test_idx in kfold.split(X):
    # Compute guest means only on training data
    train_guest_means = compute_guest_means(y[train_idx], guests[train_idx])
    # Apply to both train and test
    X_train_encoded = encode_guests(guests[train_idx], train_guest_means)
    X_test_encoded = encode_guests(guests[test_idx], train_guest_means)
```

### Hardware Requirements
- **Training**: CPU sufficient, ~2GB RAM
- **Inference**: CPU sufficient, ~1GB RAM  
- **GPU**: Optional (MPS on Apple Silicon supported)
- **Processing Speed**: ~50 chunks per second on M1 MacBook Pro

---

*This project demonstrates how modern NLP techniques can automate content curation tasks that traditionally required significant manual effort, achieving near-human performance in identifying engaging content segments.*
