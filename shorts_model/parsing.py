import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


def parse_timestamp(ts: str) -> Optional[float]:
    """Parse timestamps like M:SS or H:MM:SS into seconds (float)."""
    if ts is None:
        return None
    parts = ts.strip().split(":")
    try:
        parts = [int(p) for p in parts]
    except ValueError:
        return None
    if len(parts) == 2:
        m, s = parts
        return m * 60 + s
    if len(parts) == 3:
        h, m, s = parts
        return h * 3600 + m * 60 + s
    return None


SPEAKER_TS_RE = re.compile(r"^(?P<speaker>[A-Za-z .\-']+)\s*\((?P<ts>\d{1,2}:\d{2}(?::\d{2})?)\):\s*$")
TS_ONLY_RE = re.compile(r"^\((?P<ts>\d{1,2}:\d{2}(?::\d{2})?)\):\s*$")
LINE_NUM_PREFIX_RE = re.compile(r"^\d+\|\s*")
PART_MARKER_RE = re.compile(r"^PART\s+\d+\s+OF\s+\d+\s+(STARTS|ENDS)", re.IGNORECASE)


@dataclass
class Utterance:
    speaker: Optional[str]
    start_time: Optional[float]
    end_time: Optional[float]
    text: str


def normalize_line(line: str) -> str:
    # Remove leading line-number prefixes like "123|"
    line = LINE_NUM_PREFIX_RE.sub("", line)
    return line.rstrip("\n")


def parse_transcript_lines(lines: List[str]) -> List[Utterance]:
    """Parse transcript lines into utterances using speaker/timestamp markers."""
    utterances: List[Utterance] = []
    cur_speaker: Optional[str] = None
    cur_start: Optional[float] = None
    buffer: List[str] = []

    def flush(next_start: Optional[float]):
        nonlocal buffer, cur_speaker, cur_start
        if buffer:
            text = " ".join(x.strip() for x in buffer if x.strip())
            if text:
                utterances.append(Utterance(cur_speaker, cur_start, next_start, text))
        buffer = []

    for raw in lines:
        line = normalize_line(raw)
        if not line.strip():
            continue
        if PART_MARKER_RE.search(line):
            # Flush at hard part boundary without speaker change
            flush(next_start=None)
            continue
        m = SPEAKER_TS_RE.match(line)
        if m:
            ts = parse_timestamp(m.group("ts"))
            # New utterance boundary with speaker and timestamp
            flush(next_start=ts)
            cur_speaker = m.group("speaker").strip()
            cur_start = ts
            continue
        m2 = TS_ONLY_RE.match(line)
        if m2:
            ts = parse_timestamp(m2.group("ts"))
            # Timestamp-only boundary: end current, start new at same speaker
            flush(next_start=ts)
            cur_start = ts
            continue
        # Content line
        buffer.append(line)

    # Flush remaining
    flush(next_start=None)
    return utterances


def read_transcript(path: str) -> List[Utterance]:
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return parse_transcript_lines(lines)

