#!/usr/bin/env python3
"""
Make mixed training CSVs from real and synthetic CSVs.

Usage examples:
  # Ratios are fractions of the real row count (e.g., 0.1 => 10% of len(real))
  python3 shorts_model/make_training_mixes.py \
    --real_csv data/processed/training-data_v5.1_w-o-outlier.csv \
    --synth_csv data/interim/synth_batch_v6_top5.csv \
    --ratios 0,0.1,0.2 \
    --seed 42 \
    --outdir data/processed

  # Or specify absolute synthetic counts (overrides ratios for those counts)
  python3 shorts_model/make_training_mixes.py \
    --real_csv data/processed/training-data_v5.1_w-o-outlier.csv \
    --synth_csv data/interim/synth_batch_v6_top5.csv \
    --counts 7,15,31 \
    --seed 42 \
    --outdir data/processed

Outputs:
  data/processed/training-data_mix_0.csv
  data/processed/training-data_mix_10.csv
  data/processed/training-data_mix_20.csv
  ... (and/or _abs7.csv etc.)

Notes:
- Only the four columns needed by training are written: video_id, view_count, guest_name, transcription
- Sampling is reproducible via --seed
- If requested synthetic count exceeds available, it will be clamped to len(synth)
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

REQUIRED_COLS = ["video_id", "view_count", "guest_name", "transcription"]


def load_csv_minimal(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing required columns: {missing}")
    return df[REQUIRED_COLS].copy()


def parse_float_list(s: str) -> List[float]:
    out: List[float] = []
    for part in s.split(','):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    return out


def parse_int_list(s: str) -> List[int]:
    out: List[int] = []
    for part in s.split(','):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def make_mix(real: pd.DataFrame, synth: pd.DataFrame, k_synth: int, seed: int) -> pd.DataFrame:
    if k_synth <= 0:
        return real.copy()
    k = min(k_synth, len(synth))
    sample = synth.sample(n=k, random_state=seed, replace=False)
    combo = pd.concat([real, sample], ignore_index=True)
    return combo[REQUIRED_COLS].copy()


def main():
    ap = argparse.ArgumentParser(description="Create mixed training CSVs from real and synthetic sources.")
    ap.add_argument("--real_csv", required=True, help="Path to real-only training CSV")
    ap.add_argument("--synth_csv", required=True, help="Path to synthetic pool CSV")
    ap.add_argument("--ratios", default="0,0.1,0.2", help="Comma-separated synthetic ratios relative to len(real), e.g., 0,0.1,0.2")
    ap.add_argument("--counts", default="", help="Comma-separated absolute synthetic counts (overrides ratios for those counts)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="data/processed", help="Output directory for mixed CSVs")
    ap.add_argument("--prefix", default="training-data_mix", help="Output filename prefix")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    real = load_csv_minimal(Path(args.real_csv))
    synth = load_csv_minimal(Path(args.synth_csv))

    n_real = len(real)
    ratios = [r for r in parse_float_list(args.ratios) if r >= 0]
    counts = [c for c in parse_int_list(args.counts) if c >= 0]

    # Build (label, k_synth) plan: percentage labels first, then abs labels
    plan: List[tuple[str, int]] = []
    for r in ratios:
        k = int(round(r * n_real))
        label = f"{int(round(r*100))}"
        plan.append((label, k))
    for c in counts:
        label = f"abs{c}"
        plan.append((label, c))

    # Deduplicate labels while preserving order
    seen = set()
    plan_unique: List[tuple[str, int]] = []
    for label, k in plan:
        if label in seen:
            continue
        seen.add(label)
        plan_unique.append((label, k))

    if not plan_unique:
        print("No ratios/counts specified; nothing to do.")
        return

    for label, k in plan_unique:
        mix = make_mix(real, synth, k_synth=k, seed=args.seed)
        out_path = outdir / f"{args.prefix}_{label}.csv"
        mix.to_csv(out_path, index=False)
        print(f"wrote: {out_path} | real={len(real)} synth_kept={min(k, len(synth))} total={len(mix)}")


if __name__ == "__main__":
    main()
