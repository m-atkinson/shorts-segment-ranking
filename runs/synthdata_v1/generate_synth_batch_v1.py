#!/usr/bin/env python3
import argparse
import json
import os
import uuid
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

# Add project root for imports
import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shorts_model.parsing_yaml import load_yaml_transcript, fix_mojibake
from shorts_model.chunking import chunk_utterances
from shorts_model.embedding import EmbeddingConfig, Embedder
from shorts_model.modeling import load_scorer
from shorts_model.selection import topk_with_diversity


def normalize_guest(name: str) -> Tuple[str, str]:
    g = (name or "").strip()
    if not g:
        g = "Unknown"
    slug = g.lower().strip()
    slug = "-".join(slug.split())
    slug = ''.join(ch for ch in slug if ch.isalnum() or ch == '-')
    if not slug:
        slug = "unknown"
    return g, slug


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b) / denom)


def main():
    ap = argparse.ArgumentParser(description="Generate synthetic shorts from YAML transcripts.")
    ap.add_argument("--yaml_dir", required=True, help="Directory containing YAML transcripts")
    ap.add_argument("--regressor_path", required=True, help="Path to trained regressor pickle")
    ap.add_argument("--model_name", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--target_tokens", type=int, default=220)
    ap.add_argument("--overlap", type=float, default=0.20)
    ap.add_argument("--k_total", type=int, default=10, help="Total synthetic examples to emit")
    ap.add_argument("--max_per_transcript", type=int, default=2)
    ap.add_argument("--score_percentile", type=float, default=75.0)
    ap.add_argument("--dedup_cosine", type=float, default=0.95)
    ap.add_argument("--min_tokens", type=int, default=25, help="Minimum token length for a chunk to be considered")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    ap.add_argument("--output_stem", default="runs/synthdata_v1/synth_batch_001")
    ap.add_argument("--train_cleaned_parquet", default="runs/datasets/training_data_v4_cleaned.parquet")
    args = ap.parse_args()

    # Set seeds for reproducibility
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.output_stem).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load existing cleaned training for dedup
    if Path(args.train_cleaned_parquet).exists():
        df_train = pd.read_parquet(args.train_cleaned_parquet)
        existing_texts = df_train["text"].astype(str).tolist()
    else:
        df_train = None
        existing_texts = []

    # Embed existing texts for dedup
    embedder = Embedder(EmbeddingConfig(model_name=args.model_name))
    if existing_texts:
        existing_emb = embedder.encode(existing_texts)
    else:
        # Use model dim where possible
        try:
            dim = embedder.model.get_sentence_embedding_dimension()
        except Exception:
            dim = 384
        existing_emb = np.zeros((0, dim), dtype=np.float32)

    # Prepare scorer; guest-aware models will use per-file guest
    scorer = None  # built per guest below via load_scorer

    candidates = []  # (score, source_file, guest_name, guest_slug, chunk, emb)

    # Build a map from base filename to full path for fast lookup
    path_map = {}
    yaml_files = []
    for root, _, files in os.walk(args.yaml_dir):
        for fn in files:
            if fn.lower().endswith((".yaml", ".yml")):
                full = os.path.join(root, fn)
                yaml_files.append(full)
                path_map[fn] = full

    if not yaml_files:
        print(f"No YAML files found in {args.yaml_dir}")
        return

    for ypath in yaml_files:
        text, guest_name = load_yaml_transcript(ypath)
        if not text or len(text.strip()) < 50:
            continue
        # Clean guest name
        guest_name = fix_mojibake(guest_name or "").strip() if guest_name else ""
        gname, gslug = normalize_guest(guest_name)
        # Chunk
        from shorts_model.parsing import Utterance
        utt = Utterance(speaker=gname, start_time=None, end_time=None, text=text)
        chunks = chunk_utterances([utt], source_id=Path(ypath).name, target_tokens=args.target_tokens, overlap_frac=args.overlap)
        if not chunks:
            continue
        # Embed and score with guest-aware or baseline scorer
        chunk_texts = [c.text for c in chunks]
        emb = embedder.encode(chunk_texts)
        local_scorer = load_scorer(args.regressor_path, guest_name=gname)
        scores = local_scorer.predict(emb)
        # Normalize chunk texts too (consistent with training cleaning)
        def clean_text(t: str) -> str:
            return fix_mojibake(t)
        for c, e, s in zip(chunks, emb, scores):
            c.text = clean_text(c.text)
            # Min token length filter
            if len(c.text.split()) < args.min_tokens:
                continue
            candidates.append((float(s), ypath, gname, gslug, c, e))

    if not candidates:
        print("No candidates generated.")
        return

    # Score threshold
    scores_arr = np.array([s for s, *_ in candidates])
    thresh = np.percentile(scores_arr, args.score_percentile)

    # Sort by score desc
    candidates.sort(key=lambda x: -x[0])

    selected_rows = []
    selected_embs = []
    per_source_count = {}

    # Helper: validate a candidate by comparing to source transcript
    # Cache for loaded source transcripts
    source_cache = {}
    def validate_against_source(row: dict):
        src = row.get("source_file")
        if not src:
            return False, "no_source_file"
        ypath = path_map.get(src)
        if not ypath:
            return False, "no_path"
        if ypath in source_cache:
            full_text, guest_detected = source_cache[ypath]
        else:
            full_text, guest_detected = load_yaml_transcript(ypath)
            source_cache[ypath] = (full_text, guest_detected)
        if not full_text or len(full_text) < 50:
            return False, "no_full_text"
        # Check guest
        g_out = (row.get("guest_name") or "").strip()
        if guest_detected:
            if normalize_guest(guest_detected)[0] != g_out:
                return False, "guest_mismatch"
        # Check that chunk text appears in full_text
        chunk_text = (row.get("text") or "").strip()
        if not chunk_text:
            return False, "empty_chunk"
        if chunk_text[:30].lower().startswith(("home summaries", "download pdf", "guest biographies", "support our work", "contact", "search")):
            return False, "nav_header"
        if chunk_text not in full_text:
            # lenient check: compare after removing spaces
            if "".join(chunk_text.split()) not in "".join(full_text.split()):
                return False, "not_in_source"
        return True, "ok"

    # Dedup vs existing and among selected
    def is_duplicate(e: np.ndarray) -> bool:
        # vs existing
        if existing_emb.shape[0] > 0:
            sims = existing_emb @ e / (np.linalg.norm(existing_emb, axis=1) * np.linalg.norm(e) + 1e-8)
            if float(np.max(sims)) >= args.dedup_cosine:
                return True
        # vs selected
        for ee in selected_embs:
            if cosine_sim(ee, e) >= args.dedup_cosine:
                return True
        return False

    # Rejection counters
    rejects = {
        "below_threshold": 0,
        "per_transcript_cap": 0,
        "duplicate": 0,
        "validate_no_source_file": 0,
        "validate_no_path": 0,
        "validate_no_full_text": 0,
        "validate_guest_mismatch": 0,
        "validate_empty_chunk": 0,
        "validate_nav_header": 0,
        "validate_not_in_source": 0,
        "too_short": 0,
    }

    for s, ypath, gname, gslug, c, e in candidates:
        if len(selected_rows) >= args.k_total:
            break
        if s < thresh:
            rejects["below_threshold"] += 1
            continue
        src = ypath
        cnt = per_source_count.get(src, 0)
        if cnt >= args.max_per_transcript:
            rejects["per_transcript_cap"] += 1
            continue
        if is_duplicate(e):
            rejects["duplicate"] += 1
            continue
        # Build row
        import random
        chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
        vid = "".join(random.choice(chars) for _ in range(11))
        y_log_pred = float(s)
        view_pred = max(0, int(round(np.exp(y_log_pred) - 1)))
        row = {
            "video_id": vid,
            "guest_name": gname,
            "guest": gname,
            "guest_slug": gslug,
            "transcription": c.text,
            "text": c.text,
            "y_log_pred": y_log_pred,
            "view_count_pred": view_pred,
            "is_synthetic": True,
            "source_file": Path(ypath).name,
            "chunk_id": c.chunk_id,
            "start_time": c.start_time,
            "end_time": c.end_time,
            "model_name": args.model_name,
            "regressor_path": args.regressor_path,
            "created_at": pd.Timestamp.utcnow().isoformat(),
        }
        # Validate row against source
        ok, reason = validate_against_source(row)
        if not ok:
            key = f"validate_{reason}"
            if key in rejects:
                rejects[key] += 1
            continue
        # Accept
        per_source_count[src] = cnt + 1
        selected_embs.append(e)
        selected_rows.append(row)

    if not selected_rows:
        print("No rows selected after filtering.")
        return

    df_out = pd.DataFrame(selected_rows)

    # Build training-format table matching real dataset columns
    # Columns: video_id, view_count, guest_name, transcription, text, guest, guest_slug, y_log
    # Replace empty/whitespace guest with Unknown
    guest_name_series = df_out["guest_name"].astype(str).replace(r"^\s*$", np.nan, regex=True).fillna("Unknown")
    guest_slug_series = df_out["guest_slug"].astype(str).replace(r"^\s*$", np.nan, regex=True).fillna("unknown")

    df_trainfmt = pd.DataFrame({
        "video_id": df_out["video_id"],
        "view_count": df_out["view_count_pred"].astype(int),
        "guest_name": guest_name_series,
        "transcription": df_out["text"],
        "text": df_out["text"],
        "guest": guest_name_series,
        "guest_slug": guest_slug_series,
        "y_log": df_out["y_log_pred"].astype(float),
    })

    out_parquet = Path(args.output_stem + ".parquet")
    out_csv = Path(args.output_stem + ".csv")
    out_train_parquet = Path(args.output_stem + "_training_format.parquet")
    out_train_csv = Path(args.output_stem + "_training_format.csv")

    df_out.to_parquet(out_parquet, index=False)
    df_out.to_csv(out_csv, index=False)
    df_trainfmt.to_parquet(out_train_parquet, index=False)
    df_trainfmt.to_csv(out_train_csv, index=False)

    # Write simple report
    report = Path(args.output_stem + "_report.md")
    lines = []
    lines.append("# Synthetic Batch Report\n")
    lines.append(f"YAML dir: {args.yaml_dir}")
    lines.append(f"Regressor: {args.regressor_path}")
    lines.append(f"Model: {args.model_name}")
    lines.append(f"Selected: {len(df_out)} / Candidates: {len(candidates)}")
    lines.append(f"Score threshold (p{args.score_percentile:.0f}): {thresh:.3f}")
    lines.append(f"Max per transcript: {args.max_per_transcript}")
    lines.append(f"Dedup cosine ≥ {args.dedup_cosine}")
    lines.append("")
    for i, r in df_out.iterrows():
        lines.append(f"- {r['video_id']} | guest={r['guest_name']} | score(y_log)={r['y_log_pred']:.3f} | src={r['source_file']} | chunk={int(r['chunk_id'])}")
    lines.append("")
    lines.append("## Rejection summary")
    for k, v in rejects.items():
        lines.append(f"- {k}: {v}")
    report.write_text("\n".join(lines))

    print("Saved synthetic batch:")
    print(" ", out_parquet)
    print(" ", out_csv)
    print("Training-format outputs:")
    print(" ", out_train_parquet)
    print(" ", out_train_csv)
    print(" ", report)


if __name__ == "__main__":
    main()

