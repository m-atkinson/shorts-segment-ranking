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


def pick_device() -> str:
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def embed_texts(texts, model_name: str, batch_size: int = 128, device: str = "cpu"):
    model = SentenceTransformer(model_name, device=device)
    emb = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=True,
    )
    return emb, emb.shape[1]


def cv_real_only(
    X: np.ndarray,
    y: np.ndarray,
    is_synth: np.ndarray,
    synth_weight: float = 0.5,
    folds: int = 5,
    seed: int = 42,
):
    # Indices of real samples
    real_idx = np.where(~is_synth)[0]
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    r2s, rhos = [], []
    for tr_real, te_real in kf.split(real_idx):
        tr_real_idx = real_idx[tr_real]
        te_real_idx = real_idx[te_real]
        # Training set: all synth + real training fold
        tr_idx = np.concatenate([tr_real_idx, np.where(is_synth)[0]])
        # Sample weights
        sw = np.ones(len(tr_idx), dtype=float)
        sw[len(tr_real_idx):] = synth_weight
        reg = Ridge(alpha=1.0)
        reg.fit(X[tr_idx], y[tr_idx], sample_weight=sw)
        y_pred = reg.predict(X[te_real_idx])
        r2s.append(float(reg.score(X[te_real_idx], y[te_real_idx])))
        rhos.append(float(spearmanr(y[te_real_idx], y_pred).correlation))
    return float(np.mean(r2s)), float(np.std(r2s)), float(np.mean(rhos)), float(np.std(rhos))


def main():
    ap = argparse.ArgumentParser(description="Train MiniLM Ridge on real+synth, evaluate on real-only CV")
    ap.add_argument("--real_parquet", default="runs/datasets/training_data_v4_cleaned.parquet")
    ap.add_argument("--synth_parquet", required=True, help="Synthetic training-format parquet")
    ap.add_argument("--model_name", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--synth_weight", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    device = pick_device()

    # Load real data
    real_df = pd.read_parquet(args.real_parquet)
    # Expect columns: video_id, view_count, guest_name, transcription, text, guest, guest_slug, y_log
    real_df = real_df[real_df["text"].astype(str).str.len() > 0].copy()
    real_df["is_synth"] = False
    real_df["y_log"] = real_df["y_log"].astype(float)

    # Load synthetic data (training-format)
    synth_df = pd.read_parquet(args.synth_parquet)
    synth_df = synth_df[synth_df["text"].astype(str).str.len() > 0].copy()
    synth_df["is_synth"] = True
    synth_df["y_log"] = synth_df["y_log"].astype(float)

    # Combine
    df = pd.concat([real_df, synth_df], ignore_index=True)

    texts = df["text"].astype(str).tolist()
    y = df["y_log"].values
    is_synth = df["is_synth"].values

    # Embed
    X, dim = embed_texts(texts, args.model_name, device=device)

    # CV on real-only
    r2_mean, r2_std, rho_mean, rho_std = cv_real_only(
        X, y, is_synth, synth_weight=args.synth_weight, folds=5, seed=args.seed
    )

    # Train final on full (real + synth)
    tr_idx_full = np.arange(len(df))
    sw_full = np.where(is_synth, args.synth_weight, 1.0)
    reg = Ridge(alpha=1.0)
    reg.fit(X[tr_idx_full], y[tr_idx_full], sample_weight=sw_full)

    # Save artifacts
    import pickle
    model_path = here / "ridge_regressor_synthmix_v1.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            "regressor": reg,
            "model_name": args.model_name,
            "model_dim": dim,
            "alpha": 1.0,
            "device": device,
            "synth_weight": args.synth_weight,
            "seed": args.seed,
        }, f)

    # Report
    report = here / "training_report_synthmix_v1.md"
    lines = []
    lines.append("# Regression Head Training Report — SynthMix (v1)\n")
    lines.append(f"Real: {args.real_parquet}")
    lines.append(f"Synth: {args.synth_parquet}")
    lines.append(f"Model: {args.model_name} | Dim: {dim} | Device: {device}")
    lines.append(f"Samples: total={len(df)} real={int((~is_synth).sum())} synth={int(is_synth.sum())}")
    lines.append(f"Weights: real=1.0 synth={args.synth_weight}")
    lines.append("")
    lines.append("## Real-only CV metrics (Ridge, 5-fold)\n")
    lines.append(f"- R^2 mean ± std: {r2_mean:.3f} ± {r2_std:.3f}")
    lines.append(f"- Spearman mean ± std: {rho_mean:.3f} ± {rho_std:.3f}\n")
    lines.append("## Artifact\n")
    lines.append(f"- Regressor: {model_path.name}")
    report.write_text("\n".join(lines))

    # Manifest
    manifest = {
        "run": "train_regressor_minilm_synthmix_v1",
        "real_parquet": str(args.real_parquet),
        "synth_parquet": str(args.synth_parquet),
        "model_name": args.model_name,
        "model_dim": dim,
        "alpha": 1.0,
        "device": device,
        "seed": args.seed,
        "synth_weight": args.synth_weight,
        "samples": {
            "total": int(len(df)),
            "real": int((~is_synth).sum()),
            "synth": int(is_synth.sum()),
        },
        "cv": {
            "folds": 5,
            "r2_mean": r2_mean,
            "r2_std": r2_std,
            "spearman_mean": rho_mean,
            "spearman_std": rho_std,
        },
        "artifact": str(model_path.name),
    }
    (here / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print("Training complete (synthmix v1).")
    print(f"  Regressor: {model_path}")
    print(f"  Report:    {report}")


if __name__ == "__main__":
    main()

