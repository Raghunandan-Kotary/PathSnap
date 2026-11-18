"""Metric definitions shared by every model (Person 2 owns these; see docs/INTERFACES.md).

Two levels:

* candidate level  - every (GPS point, candidate arc) row is a binary example (label 1 = arc is on
  the reference route). Macro-averages are over the two classes {0, 1}.
* observation level - for each GPS point, the candidate with the highest predicted probability is the
  prediction. Top-1 is correct when that candidate has label 1.

Route continuity and sequence-level scores need the road graph and belong to Person 3's
`evaluation/` code; the harness accepts them through `route_metrics_fn` and reports N/A until then.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (brier_score_loss, confusion_matrix, f1_score, log_loss,
                             precision_score, recall_score, roc_auc_score)

KEYS = ["trip_id", "obs_idx"]


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def candidate_metrics(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict:
    """Candidate-level classification metrics from probabilities `p` of the positive class."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype="float64")
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out = {
        "n_candidates": int(len(y)),
        "positive_rate": float(y.mean()),
        "macro_f1": float(f1_score(y, pred, average="macro", labels=[0, 1], zero_division=0)),
        "macro_precision": float(precision_score(y, pred, average="macro", labels=[0, 1], zero_division=0)),
        "macro_recall": float(recall_score(y, pred, average="macro", labels=[0, 1], zero_division=0)),
        "pos_f1": float(f1_score(y, pred, pos_label=1, zero_division=0)),
        "pos_precision": float(precision_score(y, pred, pos_label=1, zero_division=0)),
        "pos_recall": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "accuracy": float((pred == y).mean()),
        "log_loss": float(log_loss(y, np.clip(p, 1e-15, 1 - 1e-15), labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "ece": expected_calibration_error(y, p),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan"),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    return out


def top1_predictions(pred: pd.DataFrame) -> pd.DataFrame:
    """pred needs trip_id, obs_idx, arc_id, label, proba. Returns one row per observation.

    Ties on probability are broken by the smaller arc_id so the result is deterministic.
    """
    ordered = pred.sort_values(KEYS + ["proba", "arc_id"], ascending=[True, True, False, True], kind="stable")
    return ordered.drop_duplicates(KEYS, keep="first").reset_index(drop=True)


def observation_metrics(pred: pd.DataFrame, obs: pd.DataFrame | None = None) -> dict:
    """Observation-level Top-1 accuracy and candidate miss rate.

    top1_acc           correct / observations that have >= 1 candidate          (headline number)
    top1_acc_hit_only  correct / observations whose candidate set contains >= 1 positive
    candidate_miss_rate  observations with NO positive candidate / all observations in `obs`
                         (observations with zero candidates count as misses). If `obs` is None
                         the rate only covers observations that have candidates and
                         `miss_rate_scope` says so.
    """
    top1 = top1_predictions(pred)
    has_pos = pred.groupby(KEYS)["label"].max().rename("has_pos").reset_index()
    top1 = top1.merge(has_pos, on=KEYS, how="left")
    out = {
        "n_obs_with_candidates": int(len(top1)),
        "top1_acc": float(top1["label"].mean()),
        "top1_acc_hit_only": float(top1.loc[top1.has_pos == 1, "label"].mean()) if (top1.has_pos == 1).any() else float("nan"),
    }
    if obs is not None:
        trips = pred["trip_id"].unique()
        o = obs[obs.trip_id.isin(trips)]
        out["n_obs_total"] = int(len(o))
        out["candidate_miss_rate"] = float((o["n_positive"] == 0).mean())
        out["miss_rate_scope"] = "all_observations"
    else:
        out["n_obs_total"] = int(len(top1))
        out["candidate_miss_rate"] = float((top1.has_pos == 0).mean())
        out["miss_rate_scope"] = "observations_with_candidates_only"
    return out
