#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd


def slugify(name: str) -> str:
    import re
    s = (name or "").strip().lower()
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"[^a-z0-9\-]", "", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "unknown"


def main():
    ap = argparse.ArgumentParser(description="Guest-aware data prep: clean text, standardize guest name, basic EDA.")
    ap.add_argument("--csv", required=True, help="Path to training CSV with guest column")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    in_path = Path(args.csv)
    if in_path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(in_path)
    else:
        df = pd.read_csv(in_path)

    # Detect guest column
    guest_col = None
    for c in ["guest", "guest_name", "Guest", "GuestName"]:
        if c in df.columns:
            guest_col = c
            break
    if guest_col is None:
        raise ValueError("No guest column found. Expected one of: guest, guest_name, Guest, GuestName")

    # Clean text and guest
    df["text"] = df["transcription"].astype(str).str.normalize("NFKC").str.strip()
    df["guest"] = df[guest_col].astype(str).str.normalize("NFKC").str.strip()
    df["guest_slug"] = df["guest"].apply(slugify)

    # Basic label
    df["y_log"] = np.log1p(df["view_count"].astype(float))

    # Drop empties
    df = df[df["text"].str.len() > 0].copy()

    # Save cleaned parquet
    out_parquet = here / "training_guest_cleaned.parquet"
    df.to_parquet(out_parquet, index=False)

    # Build EDA report
    counts = Counter(df["guest_slug"]) 
    total = len(df)
    top = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:20]

    report = here / "dataprep_guest_report.md"
    lines = []
    lines.append("# Guest Data Prep Report\n")
    lines.append(f"Source CSV: {args.csv}")
    lines.append(f"Rows after cleaning: {total}")
    lines.append("")
    lines.append("## Top guests by count\n")
    for g, n in top:
        lines.append(f"- {g}: {n}")
    lines.append("")
    lines.append("Columns saved in parquet: video_id, transcription(text), guest, guest_slug, view_count, y_log")
    report.write_text("\n".join(lines))

    # Minimal manifest
    manifest = {
        "run": "dataprep_guest_v1",
        "source_csv": args.csv,
        "output_parquet": str(out_parquet.name),
        "rows": int(total),
        "columns": df.columns.tolist(),
    }
    (here / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print("Data prep complete.")
    print(f"  Parquet: {out_parquet}")
    print(f"  Report:  {report}")


if __name__ == "__main__":
    main()

