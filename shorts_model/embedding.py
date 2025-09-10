from dataclasses import dataclass
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class EmbeddingConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 128
    normalize_embeddings: bool = False


class Embedder:
    def __init__(self, cfg: EmbeddingConfig):
        self.cfg = cfg
        # Pick device
        device = "cpu"
        try:
            import torch
            if torch.backends.mps.is_available():
                device = "mps"
        except Exception:
            pass
        self.device = device
        self.model = SentenceTransformer(cfg.model_name, device=device)

    def encode(self, texts: List[str]) -> np.ndarray:
        emb = self.model.encode(
            texts,
            batch_size=self.cfg.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.cfg.normalize_embeddings,
            show_progress_bar=True,
        )
        return emb

