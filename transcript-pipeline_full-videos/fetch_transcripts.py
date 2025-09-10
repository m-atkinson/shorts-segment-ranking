#!/usr/bin/env python3
"""
Fetch and parse transcript pages from conversationswithbillkristol.org into YAML.

Usage examples:
  python transcript-pipeline_full-videos/fetch_transcripts.py \
    --url https://conversationswithbillkristol.org/transcript/renee-diresta-on-social-media-and-political-power/

  python transcript-pipeline_full-videos/fetch_transcripts.py \
    --urls_file urls.txt --out_dir transcript-pipeline_full-videos/transcripts

Notes:
- Writes one YAML per URL using the slug: transcript_{slug}.yaml
- YAML includes: title, date, episode, url, fetched_at, speakers, transcript.
- No external dependencies (no PyYAML). HTML parsing is heuristic.
"""

import argparse
import re
import html
import textwrap
from datetime import datetime
from pathlib import Path
from typing import List

from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError


DATE_RE = re.compile(r"([A-Z][a-z]+\s+\d{1,2},\s+\d{4})")
EP_RE = re.compile(r"(?:Episode|Ep\.?)[\s#]*(\d{1,4})", re.I)
TITLE_TAG_RE = re.compile(r"<title>(.*?)</title>", re.I | re.S)
TAG_RE = re.compile(r"<[^>]+>")
SPEAKER_RE = re.compile(r"^([A-Z][A-Za-z .\-']{1,60}?):\s*(.*)$")


def fetch_html(url: str) -> str:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0 (AgentMode)"})
    with urlopen(req, timeout=45) as r:
        return r.read().decode("utf-8", "ignore")


def parse_transcript_page(html_text: str):
    # Strip scripts/styles
    text = re.sub(r"(?is)<script.*?</script>", "", html_text)
    text = re.sub(r"(?is)<style.*?</style>", "", text)

    # Title
    m_title = TITLE_TAG_RE.search(text)
    page_title = html.unescape(m_title.group(1)).strip() if m_title else ""
    page_title = re.sub(r"\s+\|.*$", "", page_title).strip()

    # Try to narrow to transcript region
    anchor = re.search(r"(?i)Transcript", text)
    content = text[anchor.start():] if anchor else text

    # Plain text paragraphs
    plain = TAG_RE.sub("\n", content)
    plain = html.unescape(plain)
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in plain.splitlines()]
    paras = [ln for ln in lines if ln]

    # Header scan for date and episode
    header_blob = " \n ".join(paras[:300])
    m_date = DATE_RE.search(header_blob)
    m_ep = EP_RE.search(header_blob)
    date_str = m_date.group(1) if m_date else ""
    episode_num = m_ep.group(1) if m_ep else ""

    # Dialogue parse
    dialogue = []
    current_speaker = None
    buf: List[str] = []
    speakers: List[str] = []

    def flush_local(buffer: List[str], curr: str):
        if buffer:
            txt = " ".join(buffer).strip()
            if txt:
                dialogue.append({"speaker": curr or "Unknown", "text": txt})
        return []

    for p in paras:
        if p.lower().startswith("transcript"):
            continue
        m = SPEAKER_RE.match(p)
        if m:
            buf = flush_local(buf, current_speaker)
            name = m.group(1).strip()
            name_norm = " ".join(w.capitalize() for w in name.split())
            current_speaker = name_norm
            if name_norm not in speakers:
                speakers.append(name_norm)
            rest = m.group(2).strip()
            if rest:
                buf.append(rest)
        else:
            buf.append(p)
    buf = flush_local(buf, current_speaker)

    if not dialogue:
        # Fallback: single block
        dialogue = [{"speaker": "Unknown", "text": "\n\n".join(paras)}]
        speakers = ["Unknown"]

    return page_title, date_str, episode_num, speakers, dialogue


def write_yaml(out_path: Path, *, title: str, date: str, episode: str, url: str, speakers: List[str], dialogue: List[dict]):
    out_lines = []
    out_lines.append(f"title: {title}")
    out_lines.append(f"date: {date}")
    out_lines.append(f"episode: {episode}")
    out_lines.append(f"url: {url}")
    out_lines.append(f"fetched_at: {datetime.utcnow().isoformat()}Z")
    out_lines.append("speakers:")
    for sp in speakers:
        out_lines.append(f"  - {sp}")
    out_lines.append("transcript:")
    for turn in dialogue:
        out_lines.append(f"  - speaker: {turn['speaker']}")
        out_lines.append("    text: |")
        for ln in textwrap.fill(turn["text"], width=100).splitlines():
            out_lines.append(f"      {ln}")
    out_path.write_text("\n".join(out_lines), encoding="utf-8")


def slug_from_url(url: str) -> str:
    return url.rstrip('/').split('/')[-1]


def main():
    ap = argparse.ArgumentParser(description="Fetch Conversations with Bill Kristol transcripts into YAML.")
    ap.add_argument("--url", action="append", help="Transcript URL (can repeat)")
    ap.add_argument("--urls_file", type=str, help="Path to text file containing one URL per line")
    ap.add_argument("--out_dir", type=str, default="transcript-pipeline_full-videos/transcripts", help="Output directory for YAML files")
    args = ap.parse_args()

    urls: List[str] = []
    if args.url:
        urls.extend(args.url)
    if args.urls_file:
        p = Path(args.urls_file)
        if not p.exists():
            raise FileNotFoundError(f"URLs file not found: {p}")
        with p.open("r", encoding="utf-8") as fh:
            for line in fh:
                u = line.strip()
                if u:
                    urls.append(u)

    if not urls:
        ap.error("Provide at least one --url or a --urls_file")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    successes = 0
    for url in urls:
        try:
            html_text = fetch_html(url)
            title, date_str, episode, speakers, dialogue = parse_transcript_page(html_text)
            slug = slug_from_url(url)
            out_path = out_dir / f"transcript_{slug}.yaml"
            write_yaml(out_path, title=title, date=date_str, episode=episode, url=url, speakers=speakers, dialogue=dialogue)
            print(f"OK  -> {out_path}")
            successes += 1
        except (URLError, HTTPError) as e:
            print(f"ERR [{url}] network error: {e}")
        except Exception as e:
            print(f"ERR [{url}] {e}")

    print(f"Done. {successes}/{len(urls)} succeeded. Output dir: {out_dir}")


if __name__ == "__main__":
    main()

