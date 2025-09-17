#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from scipy.stats import spearmanr

# Variant B: guest feature (target encoding)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 128
SEED = 42
ALPHA = 1.0


def embed_texts(texts, model_name: str):
    device = "cpu"
    try:
        import torch
        if torch.backends.mps.is_available():
            device = "mps"
    except Exception:
        pass
    model = SentenceTransformer(model_name, device=device)
    emb = model.encode(texts, batch_size=BATCH_SIZE, convert_to_numpy=True, normalize_embeddings=False, show_progress_bar=True)
    return emb, emb.shape[1], device


def kfold_indices(n, n_splits=5, seed=42):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(kf.split(np.arange(n)))


def compute_guest_means(y: np.ndarray, guests: pd.Series):
    df = pd.DataFrame({"g": guests.values, "y": y})
    means = df.groupby("g")["y"].mean().to_dict()
    global_mean = float(df["y"].mean())
    return means, global_mean


def ndcg_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int = 5) -> float:
    """Compute NDCG@k for ranking evaluation."""
    if len(y_true) == 0:
        return 0.0
    
    k = min(k, len(y_true))
    
    # Sort by predicted scores (descending)
    order = np.argsort(y_pred)[::-1]
    y_true_sorted = y_true[order]
    
    # DCG@k
    discounts = np.log2(np.arange(k) + 2)  # +2 because log2(1) = 0
    dcg = np.sum(y_true_sorted[:k] / discounts)
    
    # IDCG@k (ideal ranking)
    y_true_ideal = np.sort(y_true)[::-1]
    idcg = np.sum(y_true_ideal[:k] / discounts)
    
    return dcg / idcg if idcg > 0 else 0.0


def map_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int = 5) -> float:
    """Compute MAP@k for ranking evaluation."""
    if len(y_true) == 0:
        return 0.0
    
    k = min(k, len(y_true))
    
    # Convert to binary relevance (above median = relevant)
    threshold = np.median(y_true)
    y_binary = (y_true >= threshold).astype(int)
    
    if np.sum(y_binary) == 0:
        return 0.0
    
    # Sort by predicted scores
    order = np.argsort(y_pred)[::-1]
    y_binary_sorted = y_binary[order]
    
    # Compute AP@k
    precisions = []
    num_relevant = 0
    for i in range(k):
        if y_binary_sorted[i] == 1:
            num_relevant += 1
            precision = num_relevant / (i + 1)
            precisions.append(precision)
    
    return np.mean(precisions) if precisions else 0.0


def recall_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int = 5) -> float:
    """Compute Recall@k for ranking evaluation."""
    if len(y_true) == 0:
        return 0.0
    
    k = min(k, len(y_true))
    
    # Convert to binary relevance (above median = relevant)
    threshold = np.median(y_true)
    y_binary = (y_true >= threshold).astype(int)
    
    total_relevant = np.sum(y_binary)
    if total_relevant == 0:
        return 0.0
    
    # Sort by predicted scores
    order = np.argsort(y_pred)[::-1]
    y_binary_sorted = y_binary[order]
    
    # Count relevant items in top-k
    relevant_in_topk = np.sum(y_binary_sorted[:k])
    
    return relevant_in_topk / total_relevant


# ... keep your imports and code above unchanged ...

def main():
    ap = argparse.ArgumentParser(description="Train MiniLM Ridge with guest feature (target encoding).")
    ap.add_argument("--csv", required=True, help="Path to training CSV with guest column")
    ap.add_argument("--outdir", default="outputs", help="Directory where outputs will be written")
    ap.add_argument("--name", default="guestfeat_v1", help="Name stem to use in output filenames (e.g., run1)")
    args = ap.parse_args()

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    in_path = Path(args.csv)
    if in_path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(in_path)
    elif in_path.suffix.lower() in [".parquet"]:
        df = pd.read_parquet(in_path)
    else:
        df = pd.read_csv(in_path)

    # Guest column detection
    guest_col = None
    for c in ["guest", "guest_name", "Guest", "GuestName"]:
        if c in df.columns:
            guest_col = c
            break
    if guest_col is None:
        raise ValueError("No guest column found.")

    # Clean text and labels
    df["text"] = df["transcription"].astype(str).str.normalize("NFKC").str.strip()
    df["guest"] = df[guest_col].astype(str).str.normalize("NFKC").str.strip().str.lower()
    df = df[df["text"].str.len() > 0].copy()
    y_log = np.log1p(df["view_count"].astype(float)).values

    # Embed
    emb, dim, device = embed_texts(df["text"].tolist(), MODEL_NAME)

    # CV with leakage-safe target encoding
    splits = kfold_indices(len(df), n_splits=5, seed=SEED)
    r2s, rhos = [], []
    ndcgs, maps, recalls = [], [], []  # Add ranking metrics
    
    for tr, te in splits:
        tr_means, tr_global = compute_guest_means(y_log[tr], df.iloc[tr]["guest"]) 
        # Build 1-D guest feature using train means
        f_tr = np.array([tr_means.get(g, tr_global) for g in df.iloc[tr]["guest"]])[:, None]
        f_te = np.array([tr_means.get(g, tr_global) for g in df.iloc[te]["guest"]])[:, None]
        X_tr = np.hstack([emb[tr], f_tr])
        X_te = np.hstack([emb[te], f_te])

        reg = Ridge(alpha=ALPHA)
        reg.fit(X_tr, y_log[tr])
        y_pred = reg.predict(X_te)
        
        # Existing regression metrics
        r2s.append(float(reg.score(X_te, y_log[te])))
        rhos.append(float(spearmanr(y_log[te], y_pred).correlation))
        
        # Add ranking metrics
        ndcgs.append(ndcg_at_k(y_log[te], y_pred, k=5))
        maps.append(map_at_k(y_log[te], y_pred, k=5))
        recalls.append(recall_at_k(y_log[te], y_pred, k=5))

    # Train final model with full guest means
    full_means, full_global = compute_guest_means(y_log, df["guest"]) 
    f_full = np.array([full_means.get(g, full_global) for g in df["guest"]])[:, None]
    X_full = np.hstack([emb, f_full])

    reg = Ridge(alpha=ALPHA)
    reg.fit(X_full, y_log)

    # Save artifacts (use --name as the stem)
    reg_path = outdir / f"ridge_regressor_{args.name}.pkl"
    with open(reg_path, "wb") as f:
        import pickle
        pickle.dump({
            "regressor": reg,
            "model_name": MODEL_NAME,
            "model_dim": dim,
            "alpha": ALPHA,
            "device": device,
            "guest_means": full_means,
            "guest_global_mean": full_global,
            "features": "embedding + guest_mean_log_views",
            "target": "y_log",
        }, f)

    report = outdir / f"training_report_{args.name}.md"
    lines = []
    lines.append("# Training Report — Guest Feature (v1)\n")
    lines.append(f"CSV: {args.csv}")
    lines.append(f"Model: {MODEL_NAME} | Dim: {dim} | Device: {device}")
    lines.append("")
    lines.append("## Cross-validated metrics (on y_log)\n")
    lines.append(f"- R^2 mean ± std: {np.mean(r2s):.3f} ± {np.std(r2s):.3f}")
    lines.append(f"- Spearman mean ± std: {np.mean(rhos):.3f} ± {np.std(rhos):.3f}")
    lines.append("")
    lines.append("## Ranking metrics (k=5)\n")
    lines.append(f"- NDCG@5 mean ± std: {np.mean(ndcgs):.3f} ± {np.std(ndcgs):.3f}")
    lines.append(f"- MAP@5 mean ± std: {np.mean(maps):.3f} ± {np.std(maps):.3f}")
    lines.append(f"- Recall@5 mean ± std: {np.mean(recalls):.3f} ± {np.std(recalls):.3f}")
    lines.append("")
    lines.append("## Artifacts\n")
    lines.append(f"- Regressor: {reg_path.name}")
    lines.append("- Guest means embedded in pickle (guest_means, guest_global_mean) for inference")
    report.write_text("\n".join(lines))

    manifest = {
        "run": "train_regressor_minilm_guestfeat_v1",
        "csv": args.csv,
        "model_name": MODEL_NAME,
        "alpha": ALPHA,
        "dim": dim,
        "device": device,
        "cv": {
            "folds": 5,
            "seed": SEED,
            "r2_mean": float(np.mean(r2s)),
            "r2_std": float(np.std(r2s)),
            "spearman_mean": float(np.mean(rhos)),
            "spearman_std": float(np.std(rhos)),
            "ndcg_at_5_mean": float(np.mean(ndcgs)),
            "ndcg_at_5_std": float(np.std(ndcgs)),
            "map_at_5_mean": float(np.mean(maps)),
            "map_at_5_std": float(np.std(maps)),
            "recall_at_5_mean": float(np.mean(recalls)),
            "recall_at_5_std": float(np.std(recalls)),
        },
        "artifacts": {
            "regressor": reg_path.name,
        }
    }
    (outdir / f"manifest_{args.name}.json").write_text(json.dumps(manifest, indent=2))

    print("Training complete (guest feature).")
    print(f"  Regressor: {reg_path}")
    print(f"  Report:    {report}")

if __name__ == "__main__":
    main()
