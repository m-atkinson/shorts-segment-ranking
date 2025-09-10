#!/usr/bin/env python3
import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from scipy.stats import spearmanr


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # chosen based on benchmark Spearman
BATCH_SIZE = 128
SEED = 42
ALPHA = 1.0  # ridge strength; can tune later


def pick_device() -> str:
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def load_training_df(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["text"] = df["transcription"].astype(str).str.normalize("NFKC").str.strip()
    df = df[df["text"].str.len() > 0].copy()
    df["label_log1p"] = np.log1p(df["view_count"].astype(float))
    return df


def embed_texts(texts, model_name: str) -> Tuple[np.ndarray, int, str]:
    device = pick_device()
    model = SentenceTransformer(model_name, device=device)
    emb = model.encode(texts, batch_size=BATCH_SIZE, convert_to_numpy=True, normalize_embeddings=False, show_progress_bar=True)
    return emb, emb.shape[1], device


def cv_metrics(X: np.ndarray, y: np.ndarray, folds: int = 5, alpha: float = 1.0, seed: int = 42):
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    r2s, rhos = [], []
    for tr, te in kf.split(X):
        reg = Ridge(alpha=alpha)
        reg.fit(X[tr], y[tr])
        y_pred = reg.predict(X[te])
        r2s.append(reg.score(X[te], y[te]))
        rhos.append(spearmanr(y[te], y_pred).correlation)
    return float(np.mean(r2s)), float(np.std(r2s)), float(np.mean(rhos)), float(np.std(rhos))


def main():
    here = Path(__file__).resolve().parent
    data_csv = here.parent.parent / "data" / "training-data.csv"

    # Load and embed
    df = load_training_df(data_csv)
    texts = df["text"].tolist()
    y = df["label_log1p"].values

    emb, dim, device = embed_texts(texts, MODEL_NAME)

    # Save embeddings next to script
    emb_path = here / f"shorts_embeddings_all-MiniLM-L6-v2_{dim}_v1.parquet"
    emb_df = pd.DataFrame({
        "video_id": df["video_id"].astype(str),
        "text": df["text"],
        "n_tokens": df["text"].str.split().apply(len),
        "view_count": df["view_count"].astype(int),
        "label_log1p": df["label_log1p"].astype(float),
        "model": MODEL_NAME,
        "model_dim": dim,
        "created_at": pd.Timestamp.utcnow().isoformat(),
        "config_version": "train_regressor_minilm_v1",
    })
    emb_df["embedding"] = [row.astype("float32") for row in emb]
    emb_df.to_parquet(emb_path, index=False)

    # CV metrics
    r2_mean, r2_std, rho_mean, rho_std = cv_metrics(emb, y, folds=5, alpha=ALPHA, seed=SEED)

    # Fit on full data and save model
    reg = Ridge(alpha=ALPHA)
    reg.fit(emb, y)
    model_path = here / "ridge_regressor_v1.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            "regressor": reg,
            "model_name": MODEL_NAME,
            "model_dim": dim,
            "alpha": ALPHA,
            "device": device,
        }, f)

    # Write metrics report
    report = here / "training_report_v1.md"
    lines = []
    lines.append("# Regression Head Training Report (v1)\n")
    lines.append(f"Model: {MODEL_NAME} | Dim: {dim} | Device: {device}\n")
    lines.append(f"Samples: {len(df)} | Target: log1p(view_count)\n")
    lines.append("")
    lines.append("## Cross-validated metrics (Ridge, 5-fold)\n")
    lines.append(f"- R^2 mean ± std: {r2_mean:.3f} ± {r2_std:.3f}")
    lines.append(f"- Spearman mean ± std: {rho_mean:.3f} ± {rho_std:.3f}\n")
    lines.append("## Artifacts\n")
    lines.append(f"- Embeddings: {emb_path.name}")
    lines.append(f"- Regressor: {model_path.name}\n")
    report.write_text("\n".join(lines))

    print("Training complete.")
    print(f"  Embeddings: {emb_path}")
    print(f"  Regressor:  {model_path}")
    print(f"  Report:     {report}")


if __name__ == "__main__":
    main()

