"""
Model (Person 3, second model): linear Support Vector Machine.
Algorithm family: kernel-based (linear kernel), margin maximisation.

Registered in person2/experiments/models.py as "linear_svm" so it trains
through the shared harness on the frozen table:

    python person2/experiments/run_model.py --model linear_svm --table <team table> --obs <obs table> \
        --features feat_dist_to_arc feat_arc_length feat_heading_diff feat_speed feat_time_gap
    ... --calibrate sigmoid --tag platt --role calibration_variant

LinearSVC only gives a signed distance to the separating hyperplane. Without
calibration, predict_proba below just squashes that margin through a logistic
function, which is not a probability (and class_weight="balanced" pushes it
towards the positive class). The HMM needs real probabilities, so the version
used as emissions is the "--calibrate sigmoid" run: the harness's trip-grouped
Platt scaling (person2/experiments/calibration.py) fits the margin -> probability
map on trips the SVM did not see. Features are standardised inside the
pipeline because an SVM is sensitive to feature scale, unlike the tree models.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


class LinearSVMScores(ClassifierMixin, BaseEstimator):
    """LinearSVC with predict_proba = logistic(margin), so it plugs into the harness and its calibration."""

    def __init__(self, C=1.0, class_weight="balanced", max_iter=5000, random_state=None):
        self.C = C
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y):
        self.svc_ = LinearSVC(C=self.C, class_weight=self.class_weight, max_iter=self.max_iter,
                              random_state=self.random_state, dual="auto").fit(X, y)
        self.classes_ = self.svc_.classes_
        return self

    def decision_function(self, X):
        return self.svc_.decision_function(X)

    def predict_proba(self, X):
        p = 1.0 / (1.0 + np.exp(-self.decision_function(X)))
        return np.column_stack([1.0 - p, p])


def build_linear_svm(params: dict, seed: int):
    return Pipeline([("scaler", StandardScaler()), ("clf", LinearSVMScores(random_state=seed, **params))])
