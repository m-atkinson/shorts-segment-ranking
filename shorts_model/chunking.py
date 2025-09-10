from dataclasses import dataclass
from typing import List, Optional, Tuple
import math
import re

from .parsing import Utterance


@dataclass
class Chunk:
    source_id: str
    chunk_id: int
    text: str
    n_tokens: int
    start_time: Optional[float]
    end_time: Optional[float]
    speakers: List[str]


def simple_sentence_split(text: str) -> List[str]:
    # A lightweight sentence splitter to avoid external deps.
    # Splits on ., !, ? followed by space/cap; keeps punctuation.
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z(])", text.strip())
    return [p.strip() for p in parts if p.strip()]


def chunk_utterances(
    utterances: List[Utterance],
    source_id: str,
    target_tokens: int = 220,
    overlap_frac: float = 0.2,
) -> List[Chunk]:
    """Create overlapping chunks (~target_tokens, 20% overlap by default)."""
    # Build a flat list of (sentence, tokens, speaker, start_time, end_time)
    sents: List[Tuple[str, int, Optional[str], Optional[float], Optional[float]]] = []
    for utt in utterances:
        for sent in simple_sentence_split(utt.text):
            tok_count = len(sent.split())
            if tok_count == 0:
                continue
            sents.append((sent, tok_count, utt.speaker, utt.start_time, utt.end_time))

    chunks: List[Chunk] = []
    if not sents:
        return chunks

    # Sliding window over sentences by approximate token budget
    i = 0
    stride_tokens = max(1, int(target_tokens * (1 - overlap_frac)))
    cid = 0
    while i < len(sents):
        cur_tokens = 0
        j = i
        speakers = set()
        while j < len(sents) and cur_tokens < target_tokens:
            cur_tokens += sents[j][1]
            if sents[j][2]:
                speakers.add(sents[j][2])
            j += 1
        # If no progress (a single very long sentence), force include one
        if j == i:
            j = min(i + 1, len(sents))
            cur_tokens = sents[i][1]
            if sents[i][2]:
                speakers.add(sents[i][2])
        text = " ".join(s[0] for s in sents[i:j])
        start_time = sents[i][3]
        end_time = sents[j - 1][4]
        chunks.append(
            Chunk(
                source_id=source_id,
                chunk_id=cid,
                text=text,
                n_tokens=cur_tokens,
                start_time=start_time,
                end_time=end_time,
                speakers=sorted(speakers),
            )
        )
        cid += 1
        # Advance by stride measured in tokens
        adv_tokens = 0
        k = i
        while k < j and adv_tokens < stride_tokens:
            adv_tokens += sents[k][1]
            k += 1
        if k == i:
            k = i + 1
        i = k

    return chunks

