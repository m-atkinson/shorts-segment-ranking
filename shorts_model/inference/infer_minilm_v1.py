#!/usr/bin/env python3
from pathlib import Path
import argparse
import json
import sys
import pandas as pd
import numpy as np

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shorts_model.parsing import read_transcript
from shorts_model.chunking import chunk_utterances
from shorts_model.embedding import EmbeddingConfig, Embedder
from shorts_model.modeling import load_scorer
from shorts_model.selection import topk_with_diversity


def main():
    ap = argparse.ArgumentParser(description="Infer top-k shorts from a long transcript.")
    ap.add_argument("--transcript", default="data/transcript_anne-applebaum.txt", help="Path to transcript file")
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--target_tokens", type=int, default=220)
    ap.add_argument("--overlap", type=float, default=0.20)
    ap.add_argument("--sim_threshold", type=float, default=0.85)
    ap.add_argument("--model_name", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--regressor_path", default="runs/train_regressor_minilm_v1/ridge_regressor_v1.pkl")
    ap.add_argument("--guest", default="", help="Guest name for guest-aware models (optional)")
    ap.add_argument("--outdir", default="inference", help="Directory to write inference outputs (reports, JSONL)")
    args = ap.parse_args()

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    # Parse and chunk
    utterances = read_transcript(args.transcript)
    source_id = Path(args.transcript).name
    chunks = chunk_utterances(utterances, source_id=source_id, target_tokens=args.target_tokens, overlap_frac=args.overlap)

    # Embed
    embedder = Embedder(EmbeddingConfig(model_name=args.model_name))
    embeddings = embedder.encode([c.text for c in chunks])

    # Score (auto-detect guest-aware models and apply --guest if needed)
    scorer = load_scorer(args.regressor_path, guest_name=args.guest)
    scores = scorer.predict(embeddings)

    # Select top-k with diversity
    selected = topk_with_diversity(chunks, embeddings, scores, k=args.top_k, sim_threshold=args.sim_threshold)

    # Write report and JSONL of all chunks
    tstem = Path(args.transcript).stem
    report_md = outdir / f"inference_report_{tstem}_top5.md"
    lines = []
    lines.append("# Inference Report (v1)\n")
    lines.append(f"Transcript: {args.transcript}")
    lines.append(f"Model: {args.model_name}")
    lines.append(f"Regressor: {args.regressor_path}")
    lines.append(f"Chunks: {len(chunks)} | Target tokens: {args.target_tokens} | Overlap: {args.overlap}")
    lines.append("")
    lines.append("## Top candidates\n")
    for rank, (ch, sc) in enumerate(selected, start=1):
        st = f"{ch.start_time:.1f}s" if ch.start_time is not None else "N/A"
        et = f"{ch.end_time:.1f}s" if ch.end_time is not None else "N/A"
        preview = ch.text[:500].replace("\n", " ")
        lines.append(f"### #{rank} | score={sc:.3f} | start={st} | end={et} | tokens={ch.n_tokens}")
        if ch.speakers:
            lines.append(f"Speakers: {', '.join(ch.speakers)}")
        lines.append("")
        lines.append(preview + ("..." if len(ch.text) > 500 else ""))
        lines.append("")
    report_md.write_text("\n".join(lines))

    all_jsonl = outdir / f"all_chunks_{tstem}.jsonl"
    with all_jsonl.open("w", encoding="utf-8") as f:
        for ch, sc in zip(chunks, scores):
            obj = {
                "source_id": ch.source_id,
                "chunk_id": ch.chunk_id,
                "text": ch.text,
                "n_tokens": ch.n_tokens,
                "start_time": ch.start_time,
                "end_time": ch.end_time,
                "speakers": ch.speakers,
                "score": float(sc),
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Wrote report: {report_md}")
    print(f"Wrote all chunks: {all_jsonl}")


if __name__ == "__main__":
    main()

