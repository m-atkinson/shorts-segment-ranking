import pickle
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class ScoringConfig:
    regressor_path: str


def _normalize_guest(name: Optional[str]) -> str:
    if not name:
        return ""
    return str(name).strip().lower()


class RidgeRegressorScorer:
    """Baseline scorer: plain regression on embeddings only."""
    def __init__(self, regressor_path: str):
        with open(regressor_path, "rb") as f:
            obj = pickle.load(f)
        self.reg = obj["regressor"]
        self.model_name = obj.get("model_name")
        self.model_dim = obj.get("model_dim")
        self.alpha = obj.get("alpha")

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        return self.reg.predict(embeddings)


class GuestFeatureScorer:
    """Scorer for models trained with an extra 1-D guest feature (target encoding).
    Expects pickle keys: regressor, guest_means, guest_global_mean.
    """
    def __init__(self, regressor_path: str, guest_name: Optional[str]):
        with open(regressor_path, "rb") as f:
            obj = pickle.load(f)
        self.reg = obj["regressor"]
        self.model_name = obj.get("model_name")
        self.model_dim = obj.get("model_dim")
        self.alpha = obj.get("alpha")
        self.guest_means = obj.get("guest_means", {})
        self.guest_global_mean = float(obj.get("guest_global_mean", 0.0))
        self.guest = _normalize_guest(guest_name)

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        # Append the per-guest mean as a single feature to each row
        val = float(self.guest_means.get(self.guest, self.guest_global_mean))
        f = np.full((embeddings.shape[0], 1), val, dtype=embeddings.dtype)
        X = np.hstack([embeddings, f])
        return self.reg.predict(X)


class GuestNormScorer:
    """Scorer for models trained on residual target (y_log - guest_mean_log).
    Expects pickle keys: regressor, guest_means, guest_global_mean, target='y_log_residual'.
    """
    def __init__(self, regressor_path: str, guest_name: Optional[str]):
        with open(regressor_path, "rb") as f:
            obj = pickle.load(f)
        self.reg = obj["regressor"]
        self.model_name = obj.get("model_name")
        self.model_dim = obj.get("model_dim")
        self.alpha = obj.get("alpha")
        self.guest_means = obj.get("guest_means", {})
        self.guest_global_mean = float(obj.get("guest_global_mean", 0.0))
        self.guest = _normalize_guest(guest_name)

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        # Predict residuals and add back guest mean baseline
        resid = self.reg.predict(embeddings)
        base = float(self.guest_means.get(self.guest, self.guest_global_mean))
        return resid + base


def load_scorer(regressor_path: str, guest_name: Optional[str] = None):
    """Factory that returns an appropriate scorer based on pickle metadata."""
    with open(regressor_path, "rb") as f:
        obj = pickle.load(f)
    target = obj.get("target")
    features = obj.get("features", "")
    # Prefer explicit target markers
    if target == "y_log_residual":
        return GuestNormScorer(regressor_path, guest_name)
    # If features mention guest, assume guest feature model
    if isinstance(features, str) and "guest" in features:
        return GuestFeatureScorer(regressor_path, guest_name)
    # Fallback: plain ridge on embeddings
    return RidgeRegressorScorer(regressor_path)

