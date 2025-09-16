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


def main():
    ap = argparse.ArgumentParser(description="Train MiniLM Ridge with guest feature (target encoding).")
    ap.add_argument("--csv", required=True, help="Path to training CSV with guest column")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
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
        r2s.append(float(reg.score(X_te, y_log[te])))
        rhos.append(float(spearmanr(y_log[te], y_pred).correlation))

    # Train final model with full guest means
    full_means, full_global = compute_guest_means(y_log, df["guest"]) 
    f_full = np.array([full_means.get(g, full_global) for g in df["guest"]])[:, None]
    X_full = np.hstack([emb, f_full])

    reg = Ridge(alpha=ALPHA)
    reg.fit(X_full, y_log)

    # Save artifacts
    reg_path = here / "ridge_regressor_guestfeat_v1.pkl"
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

    report = here / "training_report_guestfeat_v1.md"
    lines = []
    lines.append("# Training Report — Guest Feature (v1)\n")
    lines.append(f"CSV: {args.csv}")
    lines.append(f"Model: {MODEL_NAME} | Dim: {dim} | Device: {device}")
    lines.append("")
    lines.append("## Cross-validated metrics (on y_log)\n")
    lines.append(f"- R^2 mean ± std: {np.mean(r2s):.3f} ± {np.std(r2s):.3f}")
    lines.append(f"- Spearman mean ± std: {np.mean(rhos):.3f} ± {np.std(rhos):.3f}")
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
        },
        "artifacts": {
            "regressor": reg_path.name,
        }
    }
    (here / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print("Training complete (guest feature).")
    print(f"  Regressor: {reg_path}")
    print(f"  Report:    {report}")


if __name__ == "__main__":
    main()

