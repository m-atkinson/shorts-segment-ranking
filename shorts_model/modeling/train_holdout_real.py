#!/usr/bin/env python3
"""
Train on real+synthetic while holding out a subset of real-only rows for evaluation.

- Train set: all rows from --train_csv EXCEPT a held-out fraction of rows whose
  video_id appears in --real_csv (treated as the pool of real examples).
- Test set: the held-out real rows (unseen during training).

Evaluates:
- Regression metrics on the holdout (R^2, Spearman)
- Ranking metrics (NDCG@5, MAP@5, Recall@5) averaged over guests

Outputs:
- Trained regressor pickle
- Markdown report with metrics
- JSON manifest with details
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

# Local copies of training utilities to avoid package import issues
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
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, device=device)
    emb = model.encode(texts, batch_size=BATCH_SIZE, convert_to_numpy=True, normalize_embeddings=False, show_progress_bar=True)
    return emb, emb.shape[1], device


def compute_guest_means(y: np.ndarray, guests: pd.Series):
    df = pd.DataFrame({"g": guests.values, "y": y})
    means = df.groupby("g")["y"].mean().to_dict()
    global_mean = float(df["y"].mean())
    return means, global_mean


def ndcg_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int = 5) -> float:
    if len(y_true) == 0:
        return 0.0
    k = min(k, len(y_true))
    order = np.argsort(y_pred)[::-1]
    y_true_sorted = y_true[order]
    discounts = np.log2(np.arange(k) + 2)
    dcg = float(np.sum(y_true_sorted[:k] / discounts))
    y_true_ideal = np.sort(y_true)[::-1]
    idcg = float(np.sum(y_true_ideal[:k] / discounts))
    return dcg / idcg if idcg > 0 else 0.0


def map_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int = 5) -> float:
    if len(y_true) == 0:
        return 0.0
    k = min(k, len(y_true))
    threshold = np.median(y_true)
    y_binary = (y_true >= threshold).astype(int)
    if int(np.sum(y_binary)) == 0:
        return 0.0
    order = np.argsort(y_pred)[::-1]
    yb = y_binary[order]
    precisions = []
    num_rel = 0
    for i in range(k):
        if yb[i] == 1:
            num_rel += 1
            precisions.append(num_rel / (i + 1))
    return float(np.mean(precisions)) if precisions else 0.0


def recall_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int = 5) -> float:
    if len(y_true) == 0:
        return 0.0
    k = min(k, len(y_true))
    threshold = np.median(y_true)
    y_binary = (y_true >= threshold).astype(int)
    total_rel = int(np.sum(y_binary))
    if total_rel == 0:
        return 0.0
    order = np.argsort(y_pred)[::-1]
    yb = y_binary[order]
    rel_topk = int(np.sum(yb[:k]))
    return rel_topk / total_rel


def group_by_guest_metrics(y_true: np.ndarray, y_pred: np.ndarray, guests: pd.Series, k: int = 5) -> Dict[str, float]:
    """Compute ranking metrics per guest and average."""
    df = pd.DataFrame({"y": y_true, "p": y_pred, "g": guests.values})
    ndcgs, maps, recalls = [], [], []
    for g, sub in df.groupby("g"):
        if len(sub) < 2:
            # Need at least 2 items to form a ranking
            continue
        y = sub["y"].values
        p = sub["p"].values
        ndcgs.append(ndcg_at_k(y, p, k))
        maps.append(map_at_k(y, p, k))
        recalls.append(recall_at_k(y, p, k))
    out = {
        "ndcg_at_5_mean": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "ndcg_at_5_std": float(np.std(ndcgs)) if ndcgs else 0.0,
        "map_at_5_mean": float(np.mean(maps)) if maps else 0.0,
        "map_at_5_std": float(np.std(maps)) if maps else 0.0,
        "recall_at_5_mean": float(np.mean(recalls)) if recalls else 0.0,
        "recall_at_5_std": float(np.std(recalls)) if recalls else 0.0,
        "n_guests_evaluated": int(len(ndcgs)),
    }
    return out


def main():
    ap = argparse.ArgumentParser(description="Train with real-only holdout evaluation")
    ap.add_argument("--train_csv", required=True, help="CSV with real+synthetic (e.g., v4.3_with-pseudo)")
    ap.add_argument("--real_csv", required=True, help="CSV with real-only rows (e.g., v4.1_w-o-outlier)")
    ap.add_argument("--holdout_frac", type=float, default=0.2, help="Fraction of real rows to hold out")
    ap.add_argument("--outdir", default="runs/v6", help="Output directory")
    ap.add_argument("--name", default="v6_holdout_real", help="Run name")
    args = ap.parse_args()

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    # Load data
    df_all = pd.read_csv(args.train_csv)
    df_real = pd.read_csv(args.real_csv)

    # Identify real rows in the combined CSV by video_id intersection
    real_ids = set(df_real["video_id"].astype(str))
    df_all["video_id"] = df_all["video_id"].astype(str)
    is_real = df_all["video_id"].isin(real_ids)

    # Make a reproducible holdout of real-only rows
    rng = np.random.default_rng(SEED)
    real_idx = np.flatnonzero(is_real.values)
    n_holdout = max(1, int(len(real_idx) * args.holdout_frac))
    holdout_idx = set(rng.choice(real_idx, size=n_holdout, replace=False))

    # Split into train/test
    mask_holdout = df_all.index.isin(holdout_idx)
    df_test = df_all.loc[mask_holdout].copy()
    df_train = df_all.loc[~mask_holdout].copy()  # includes synthetic and remaining real

    # Clean text/guest
    for d in (df_train, df_test):
        d["text"] = d["transcription"].astype(str).str.normalize("NFKC").str.strip()
        d["guest"] = d["guest_name"].astype(str).str.normalize("NFKC").str.strip().str.lower()

    y_train = np.log1p(df_train["view_count"].astype(float)).values
    y_test = np.log1p(df_test["view_count"].astype(float)).values

    # Embed
    emb_train, dim, device = embed_texts(df_train["text"].tolist(), MODEL_NAME)
    emb_test, _, _ = embed_texts(df_test["text"].tolist(), MODEL_NAME)

    # Guest means from TRAIN ONLY (leakage-safe)
    means, global_mean = compute_guest_means(y_train, df_train["guest"])
    f_train = np.array([means.get(g, global_mean) for g in df_train["guest"]])[:, None]
    f_test = np.array([means.get(g, global_mean) for g in df_test["guest"]])[:, None]

    X_train = np.hstack([emb_train, f_train])
    X_test = np.hstack([emb_test, f_test])

    # Train
    reg = Ridge(alpha=ALPHA, random_state=SEED)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    # Metrics (global)
    ss_res = float(np.sum((y_test - y_pred) ** 2))
    ss_tot = float(np.sum((y_test - np.mean(y_test)) ** 2))
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    rho = float(spearmanr(y_test, y_pred).correlation)

    # Ranking metrics averaged by guest on the holdout
    rank = group_by_guest_metrics(y_test, y_pred, df_test["guest"], k=5)

    # Save model
    model_path = outdir / f"ridge_regressor_{args.name}.pkl"
    with open(model_path, "wb") as f:
        import pickle
        pickle.dump({
            "regressor": reg,
            "model_name": MODEL_NAME,
            "model_dim": dim,
            "alpha": ALPHA,
            "device": device,
            "guest_means": means,
            "guest_global_mean": global_mean,
            "features": "embedding + guest_mean_log_views",
            "target": "y_log",
        }, f)

    # Report
    report = outdir / f"training_report_{args.name}.md"
    lines: List[str] = []
    lines.append(f"# Real-Only Holdout Evaluation — {args.name}\n")
    lines.append(f"Train CSV: {args.train_csv}")
    lines.append(f"Real CSV (pool): {args.real_csv}")
    lines.append(f"Holdout fraction (of real): {args.holdout_frac}")
    lines.append("")
    lines.append(f"Train size: {len(df_train)} (real={int(df_train['video_id'].isin(real_ids).sum())}, synth/other={int((~df_train['video_id'].isin(real_ids)).sum())})")
    lines.append(f"Test size (real only): {len(df_test)}")
    lines.append("")
    lines.append("## Global metrics on holdout (real-only)\n")
    lines.append(f"- R^2: {r2:.3f}")
    lines.append(f"- Spearman: {rho:.3f}")
    lines.append("")
    lines.append("## Ranking metrics on holdout (averaged by guest, k=5)\n")
    lines.append(f"- NDCG@5: {rank['ndcg_at_5_mean']:.3f} ± {rank['ndcg_at_5_std']:.3f}")
    lines.append(f"- MAP@5: {rank['map_at_5_mean']:.3f} ± {rank['map_at_5_std']:.3f}")
    lines.append(f"- Recall@5: {rank['recall_at_5_mean']:.3f} ± {rank['recall_at_5_std']:.3f}")
    lines.append(f"- Guests evaluated: {rank['n_guests_evaluated']}")
    report.write_text("\n".join(lines))

    # Manifest
    manifest = {
        "run": args.name,
        "train_csv": args.train_csv,
        "real_csv_pool": args.real_csv,
        "holdout_frac": args.holdout_frac,
        "sizes": {
            "train": int(len(df_train)),
            "test": int(len(df_test)),
        },
        "global_metrics": {
            "r2": float(r2),
            "spearman": float(rho),
        },
        "ranking_metrics": rank,
        "artifacts": {
            "regressor": model_path.name,
            "report": report.name,
        },
    }
    (outdir / f"manifest_{args.name}.json").write_text(json.dumps(manifest, indent=2))

    print("Training complete with real-only holdout.")
    print(f"  Model:   {model_path}")
    print(f"  Report:  {report}")


if __name__ == "__main__":
    main()