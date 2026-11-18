"""Model loader for the API (Person 4).

    from person2.models.load import load_model
    model = load_model()                      # models/final/model.joblib
    p = model.predict_proba(feature_df)       # 1-D array, P(candidate is on the route), same row order as feature_df

`feature_df` is the candidate-level feature table produced by the shared feature function. Only the
columns listed in `model.feature_cols` are read (in the saved order); anything else, including a
`label` column, is ignored, so ground truth cannot influence a prediction.

Security note: joblib files are pickles. Only load model files from this repository.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

DEFAULT_DIR = Path(__file__).resolve().parent / "final"


class PathSnapModel:
    def __init__(self, bundle: dict, meta: dict | None = None):
        self._model = bundle["model"]
        self.feature_cols: list[str] = list(bundle["feature_cols"])
        self.threshold: float = float(bundle.get("threshold", 0.5))
        self.params: dict = bundle.get("params", {})
        self.calibration = bundle.get("calibration")
        self.meta = meta or {}

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        missing = [c for c in self.feature_cols if c not in features.columns]
        if missing:
            raise KeyError(f"feature table is missing columns the model was trained on: {missing}")
        X = features[self.feature_cols].to_numpy(dtype="float32")
        if np.isnan(X).any():
            raise ValueError("feature table contains NaN; the model was trained on complete features")
        return self._model.predict_proba(X)[:, 1]

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(features) >= self.threshold).astype(int)


def load_model(path: str | Path | None = None) -> PathSnapModel:
    """`path` is a model.joblib file or a folder containing one (default: models/final)."""
    p = Path(path) if path else DEFAULT_DIR
    f = p / "model.joblib" if p.is_dir() else p
    if not f.exists():
        raise FileNotFoundError(f"{f} not found. Run models/save_final.py after the final model has been selected.")
    cfg = f.with_name("config.json")
    meta = json.loads(cfg.read_text()) if cfg.exists() else {}
    return PathSnapModel(joblib.load(f), meta)


_CACHE: dict[str, PathSnapModel] = {}


def predict_proba(features: pd.DataFrame, path: str | Path | None = None) -> np.ndarray:
    """Convenience wrapper with a per-path cache, for the API's hot path."""
    key = str(path or DEFAULT_DIR)
    if key not in _CACHE:
        _CACHE[key] = load_model(path)
    return _CACHE[key].predict_proba(features)
