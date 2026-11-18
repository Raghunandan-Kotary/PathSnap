"""Leak-free, trip-grouped probability calibration.

The HMM emission model consumes these probabilities, so they should mean what they say.

`GroupCalibrated` cross-fits: the base model is trained on K-1 trip groups and scored on the held-out
group; the pooled out-of-fold scores train ONE calibrator (isotonic or Platt/sigmoid). The final base
model is then refit on all rows and its scores are passed through the calibrator. Calibration data
is therefore never data the base model saw, and trips are never split across the two roles.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

EPS = 1e-6  # keeps log-probabilities finite for the Viterbi decoder


class GroupCalibrated(ClassifierMixin, BaseEstimator):
    def __init__(self, base_estimator, method: str = "isotonic", n_splits: int = 3):
        self.base_estimator = base_estimator
        self.method = method
        self.n_splits = n_splits

    def fit(self, X, y, groups):
        X, y, groups = np.asarray(X), np.asarray(y), np.asarray(groups)
        oof = np.zeros(len(y), dtype="float64")
        n_groups = len(np.unique(groups))
        for tr, va in GroupKFold(n_splits=min(self.n_splits, n_groups)).split(X, y, groups):
            m = clone(self.base_estimator).fit(X[tr], y[tr])
            oof[va] = m.predict_proba(X[va])[:, 1]
        if self.method == "isotonic":
            self.calibrator_ = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(oof, y)
        elif self.method == "sigmoid":
            z = self._logit(oof)[:, None]
            self.calibrator_ = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(z, y)
        else:
            raise ValueError(f"unknown calibration method {self.method!r}")
        self.base_ = clone(self.base_estimator).fit(X, y)
        self.classes_ = np.array([0, 1])
        return self

    @staticmethod
    def _logit(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    def predict_proba(self, X):
        s = self.base_.predict_proba(np.asarray(X))[:, 1]
        if self.method == "isotonic":
            p = self.calibrator_.predict(s)
        else:
            p = self.calibrator_.predict_proba(self._logit(s)[:, None])[:, 1]
        p = np.clip(p, EPS, 1 - EPS)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
