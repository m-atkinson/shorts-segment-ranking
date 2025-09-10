#!/usr/bin/env python3
import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from scipy.stats import spearmanr


@dataclass
class ModelSpec:
    name: str


def pick_device() -> str:
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def model_slug(model_name: str) -> str:
    return model_name.split("/")[-1]


def load_training_df(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["text"] = df["transcription"].astype(str).str.normalize("NFKC").str.strip()
    df = df[df["text"].str.len() > 0].copy()
    df["label_log1p"] = np.log1p(df["view_count"].astype(float))
    return df


def encode_texts(model_name: str, texts: List[str], batch_size: int = 128, normalize_embeddings: bool = False) -> Tuple[np.ndarray, str]:
    device = pick_device()
    model = SentenceTransformer(model_name, device=device)
    emb = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
        show_progress_bar=True,
    )
    return emb, device


def cv_ridge_metrics(X: np.ndarray, y: np.ndarray, folds: int = 5, alpha: float = 1.0, seed: int = 42):
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    r2s, rhos = [], []
    for tr, te in kf.split(X):
        reg = Ridge(alpha=alpha)
        reg.fit(X[tr], y[tr])
        y_pred = reg.predict(X[te])
        r2s.append(reg.score(X[te], y[te]))
        rhos.append(spearmanr(y[te], y_pred).correlation)
    return float(np.mean(r2s)), float(np.std(r2s)), float(np.mean(rhos)), float(np.std(rhos))


def write_embeddings_parquet(df: pd.DataFrame, embeddings: np.ndarray, model_name: str, out_dir: Path) -> Path:
    dim = embeddings.shape[1]
    out_df = pd.DataFrame({
        "video_id": df["video_id"].astype(str),
        "text": df["text"],
        "n_tokens": df["text"].str.split().apply(len),
        "view_count": df["view_count"].astype(int),
        "label_log1p": df["label_log1p"].astype(float),
        "model": model_name,
        "model_dim": dim,
        "created_at": pd.Timestamp.utcnow().isoformat(),
        "config_version": "benchmark_v3",
    })
    out_df["embedding"] = [e.astype("float32") for e in embeddings]

    slug = model_slug(model_name)
    out_path = out_dir / f"shorts_embeddings_{slug}_{dim}_v3.parquet"
    out_df.to_parquet(out_path, index=False)
    return out_path


def main():
    here = Path(__file__).resolve().parent
    data_csv = here.parent.parent / "data" / "training-data_v3.csv"
    report_path = here / "embedding_benchmark_v3.md"

    models = [
        ModelSpec("sentence-transformers/all-MiniLM-L6-v2"),
        ModelSpec("intfloat/e5-base-v2"),
        ModelSpec("BAAI/bge-base-en-v1.5"),
    ]

    df = load_training_df(str(data_csv))
    texts = df["text"].tolist()
    y = df["label_log1p"].values

    rows = []

    for spec in models:
        print(f"\n==== Running {spec.name} ====")
        emb, device = encode_texts(spec.name, texts, batch_size=128, normalize_embeddings=False)
        r2_mean, r2_std, rho_mean, rho_std = cv_ridge_metrics(emb, y, folds=5, alpha=1.0, seed=42)
        out_path = write_embeddings_parquet(df, emb, spec.name, out_dir=here)
        print(f"Model: {spec.name} | device={device} | dim={emb.shape[1]} | n={len(df)}")
        print(f"R2 mean={r2_mean:.3f} std={r2_std:.3f} | Spearman mean={rho_mean:.3f} std={rho_std:.3f}")
        rows.append({
            "model": spec.name,
            "device": device,
            "dim": emb.shape[1],
            "n": len(df),
            "r2_mean": r2_mean,
            "r2_std": r2_std,
            "rho_mean": rho_mean,
            "rho_std": rho_std,
            "parquet": str(out_path.relative_to(here)),
        })

    ts = pd.Timestamp.utcnow().isoformat()
    lines = []
    lines.append(f"# Embedding Benchmark v3\n")
    lines.append(f"Generated: {ts} UTC\n")
    lines.append(f"Dataset: {data_csv} | Samples: {len(df)}\n")
    lines.append("")
    lines.append("| Model | Dim | Device | N | R^2 mean ± std | Spearman mean ± std | Output |")
    lines.append("|---|---:|:---:|---:|---:|---:|---|")
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['dim']} | {r['device']} | {r['n']} | "
            f"{r['r2_mean']:.3f} ± {r['r2_std']:.3f} | {r['rho_mean']:.3f} ± {r['rho_std']:.3f} | {r['parquet']} |"
        )
    lines.append("")
    lines.append("Notes:\n- Labels use log1p(view_count).\n- Embeddings stored as float32 lists in parquet in this folder.\n- CV: Ridge, 5 folds, alpha=1.0.\n")

    report_path.write_text("\n".join(lines))
    print(f"\nWrote report: {report_path}")


if __name__ == "__main__":
    main()

