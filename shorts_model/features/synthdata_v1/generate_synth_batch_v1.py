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
ROOT = Path(__file__).resolve().parents[3]
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
    ap.add_argument("--k_total", type=int, default=0, help="Optional global cap on total selected rows; 0 = no cap")
    ap.add_argument("--per_transcript_top_k", type=int, default=5, help="How many top candidates to consider per transcript before random sampling")
    ap.add_argument("--per_transcript_keep", type=int, default=5, help="How many final samples to keep per transcript (random from top_k)")
    ap.add_argument("--diversity_sim_threshold", type=float, default=0.85, help="Cosine similarity threshold for diversity when building top_k list")
    ap.add_argument("--dedup_cosine", type=float, default=0.95)
    ap.add_argument("--min_tokens", type=int, default=25, help="Minimum token length for a chunk to be considered")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    ap.add_argument("--output_csv", required=True, help="Path to output training CSV (columns: video_id, view_count, guest_name, transcription)")
    ap.add_argument("--train_cleaned_parquet", default="runs/datasets/training_data_v4_cleaned.parquet")
    args = ap.parse_args()

    # Set seeds for reproducibility
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

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

    # Group candidates by transcript
    selected_rows = []
    selected_embs = []
    by_src = {}
    for s, ypath, gname, gslug, c, e in candidates:
        by_src.setdefault(ypath, []).append((s, ypath, gname, gslug, c, e))

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
        "duplicate": 0,
        "validate_no_source_file": 0,
        "validate_no_path": 0,
        "validate_no_full_text": 0,
        "validate_guest_mismatch": 0,
        "validate_empty_chunk": 0,
        "validate_nav_header": 0,
        "validate_not_in_source": 0,
    }

    import random
    # Per-transcript selection: build a top_k set with diversity, then randomly keep N from that set
    for ypath, items in by_src.items():
        # Unpack local lists
        scores = [it[0] for it in items]
        chunks = [it[4] for it in items]
        embs   = np.stack([it[5] for it in items], axis=0) if items else np.zeros((0, 1))
        # Get top_k with diversity on this transcript
        if len(items) == 0:
            continue
        k_local = min(args.per_transcript_top_k, len(items))
        # Use the selection helper over the local ranking
        pairs = topk_with_diversity(chunks, embs, np.array(scores), k=k_local, sim_threshold=args.diversity_sim_threshold)
        # Randomly choose keep_n from these pairs
        keep_n = min(args.per_transcript_keep, len(pairs))
        idxs = list(range(len(pairs)))
        random.shuffle(idxs)
        chosen = [pairs[i] for i in idxs[:keep_n]]
        # Materialize rows
        for ch, sc in chosen:
            # Find embedding for dedup: align by chunk_id
            try:
                pos = next(i for i, it in enumerate(items) if it[4].chunk_id == ch.chunk_id)
                e = items[pos][5]
                gname = items[pos][2]
                gslug = items[pos][3]
            except StopIteration:
                e = np.zeros((embs.shape[1],), dtype=embs.dtype) if embs.size else np.zeros((1,), dtype=np.float32)
                gname = items[0][2]
                gslug = items[0][3]
            if is_duplicate(e):
                rejects["duplicate"] += 1
                continue
            y_log_pred = float(sc)
            view_pred = max(0, int(round(np.exp(y_log_pred) - 1)))
            chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
            vid = "".join(random.choice(chars) for _ in range(11))
            row = {
                "video_id": vid,
                "guest_name": gname,
                "guest": gname,
                "guest_slug": gslug,
                "transcription": ch.text,
                "text": ch.text,
                "y_log_pred": y_log_pred,
                "view_count_pred": view_pred,
                "is_synthetic": True,
                "source_file": Path(ypath).name,
                "chunk_id": ch.chunk_id,
                "start_time": ch.start_time,
                "end_time": ch.end_time,
                "model_name": args.model_name,
                "regressor_path": args.regressor_path,
                "created_at": pd.Timestamp.utcnow().isoformat(),
            }
            ok, reason = validate_against_source(row)
            if not ok:
                key = f"validate_{reason}"
                if key in rejects:
                    rejects[key] += 1
                continue
            selected_embs.append(e)
            selected_rows.append(row)
            if args.k_total and len(selected_rows) >= args.k_total:
                break
        if args.k_total and len(selected_rows) >= args.k_total:
            break

    if not selected_rows:
        print("No rows selected after filtering.")
        return

    df_out = pd.DataFrame(selected_rows)

    # Build minimal training-format table matching train.py expectations
    # Columns: video_id, view_count, guest_name, transcription
    guest_name_series = df_out["guest_name"].astype(str).replace(r"^\s*$", np.nan, regex=True).fillna("Unknown")

    df_trainfmt = pd.DataFrame({
        "video_id": df_out["video_id"],
        "view_count": df_out["view_count_pred"].astype(int),
        "guest_name": guest_name_series,
        "transcription": df_out["text"],
    })

    # Write single CSV output
    df_trainfmt.to_csv(out_path, index=False)

    # Minimal console summary
    print("Saved synthetic training CSV:")
    print(" ", out_path)
    print(f"Selected rows: {len(df_trainfmt)} / Candidates: {len(candidates)} | per_transcript_top_k={args.per_transcript_top_k} keep={args.per_transcript_keep} diversity_sim_threshold={args.diversity_sim_threshold}")
    print("Rejections:")
    for k, v in rejects.items():
        print(f"  - {k}: {v}")


if __name__ == "__main__":
    main()

