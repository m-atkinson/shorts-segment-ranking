from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from .chunking import Chunk


def topk_with_diversity(
    chunks: List[Chunk],
    embeddings: np.ndarray,
    scores: np.ndarray,
    k: int = 5,
    sim_threshold: float = 0.85,
) -> List[Tuple[Chunk, float]]:
    """Select top-k by score with cosine-similarity-based diversity.
    Avoid adding a chunk if its cosine similarity to any selected chunk exceeds sim_threshold.
    """
    order = np.argsort(-scores)
    selected: List[int] = []
    for idx in order:
        if len(selected) >= k:
            break
        if not selected:
            selected.append(idx)
            continue
        v = embeddings[idx]
        too_similar = False
        for j in selected:
            u = embeddings[j]
            denom = (np.linalg.norm(u) * np.linalg.norm(v) + 1e-8)
            cos = float(np.dot(u, v) / denom)
            if cos >= sim_threshold:
                too_similar = True
                break
        if not too_similar:
            selected.append(idx)
    return [(chunks[i], float(scores[i])) for i in selected]

