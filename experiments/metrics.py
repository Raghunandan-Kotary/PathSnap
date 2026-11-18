"""
Comprehensive model evaluation metrics for PathSnap.
Includes observation-level Top-1 accuracy, candidate miss rate, Macro F1, Log Loss, and timing.
"""
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, log_loss, confusion_matrix, accuracy_score

def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray = None
) -> Dict[str, float]:
    """Computes binary classification metrics for candidate pairs."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    if y_prob is not None:
        try:
            # Clip for numerical stability
            y_prob_clipped = np.clip(y_prob, 1e-15, 1 - 1e-15)
            metrics["log_loss"] = float(log_loss(y_true, y_prob_clipped))
        except Exception:
            metrics["log_loss"] = float("nan")
    return metrics

def compute_observation_top1_accuracy(
    candidates_df: pd.DataFrame,
    probs: np.ndarray
) -> float:
    """
    Observation-level Top-1 Accuracy:
    For each GPS observation (obs_idx), picks the candidate with the highest predicted score.
    Returns fraction of observations where the top candidate is the true match (label == 1).
    """
    df = candidates_df.copy()
    df["predicted_prob"] = probs
    
    # Group by observation and pick highest scoring candidate
    idx_max = df.groupby(["trip_id", "obs_idx"])["predicted_prob"].idxmax()
    top_candidates = df.loc[idx_max]
    
    top1_acc = float((top_candidates["label"] == 1).mean())
    return top1_acc
