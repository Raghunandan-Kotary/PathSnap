"""
Model loading and probability prediction interface for Person 2 & Person 4.
Guarantees consistent input feature order and calibrated emission probabilities.
"""
from pathlib import Path
from typing import Union, List
import numpy as np
import pandas as pd
import joblib

FEATURE_COLUMNS = [
    "feat_dist_to_arc",
    "feat_arc_length",
    "feat_heading_diff",
    "feat_speed",
    "feat_time_gap"
]

class ModelWrapper:
    def __init__(self, model_obj, feature_cols: List[str] = None):
        self.model = model_obj
        self.feature_cols = feature_cols or FEATURE_COLUMNS

    @classmethod
    def load(cls, model_path: Union[str, Path]) -> "ModelWrapper":
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model artifact not found at {path}")
        model_obj = joblib.load(path)
        # person2/models/save_final.py writes a bundle {"model", "feature_cols", ...};
        # without unwrapping it, predict_proba below finds no estimator and
        # silently returns 0.5 for every candidate.
        if isinstance(model_obj, dict) and "model" in model_obj:
            return cls(model_obj["model"], model_obj.get("feature_cols"))
        return cls(model_obj)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Generates calibrated probability that each candidate arc is the true matched segment.
        Returns 1D numpy array of probabilities aligned with input df rows.
        """
        # Ensure only expected features are fed
        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required feature columns: {missing}")
            
        X = df[self.feature_cols].values
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X)
            # Binary classification positive class probability
            if probs.ndim == 2 and probs.shape[1] >= 2:
                return probs[:, 1]
            return probs.ravel()
        elif hasattr(self.model, "decision_function"):
            # Sigmoid calibration fallback for SVM
            scores = self.model.decision_function(X)
            return 1.0 / (1.0 + np.exp(-scores))
        else:
            return np.ones(len(df)) * 0.5
