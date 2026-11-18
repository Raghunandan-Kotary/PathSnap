"""Model registry for Person 2's two models. Initial values here are the ones reported in the report's
"important hyperparameters and their initial values" answer; tuned values live in tuning/.

Preprocessing lives inside each Pipeline so the harness fits it on training folds only.
"""
from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from person2.experiments.harness import ModelSpec


def build_xgboost(params: dict, seed: int):
    clf = XGBClassifier(objective="binary:logistic", eval_metric="logloss", tree_method="hist",
                        random_state=seed, n_jobs=-1, verbosity=0, **params)
    return Pipeline([("scaler", "passthrough"), ("clf", clf)])  # trees are scale-invariant


def build_gaussian_nb(params: dict, seed: int):
    # GNB is per-feature scale-invariant (fits a mean/variance per feature), but we keep a scaler
    # so the saved preprocessing object has the same shape as every other model's.
    return Pipeline([("scaler", StandardScaler()), ("clf", GaussianNB(**params))])


XGBOOST = ModelSpec(
    name="xgboost",
    family="Boosting (gradient-boosted decision trees)",
    build=build_xgboost,
    defaults=dict(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                  min_child_weight=1, reg_lambda=1.0, gamma=0.0),
)

GAUSSIAN_NB = ModelSpec(
    name="gaussian_nb",
    family="Probabilistic (generative, Naive Bayes)",
    build=build_gaussian_nb,
    defaults=dict(var_smoothing=1e-9),
)

# ---------------------------------------------------------------------------------------------
# Teammates' models, with the hyperparameters used in their own scripts (experiments/train_model*.py
# in the team repo), so the consolidated comparison (report 3.1) measures every model the same way:
# the same frozen table, the same trip-grouped folds, FULL (un-resampled) validation and test sets.
# Each owner remains responsible for explaining their model.
# ---------------------------------------------------------------------------------------------
def build_logreg(params: dict, seed: int):
    return Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(random_state=seed, **params))])


def build_knn(params: dict, seed: int):
    return Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier(n_jobs=-1, **params))])


def build_rf(params: dict, seed: int):
    return Pipeline([("scaler", "passthrough"), ("clf", RandomForestClassifier(random_state=seed, n_jobs=-1, **params))])


def build_hgb(params: dict, seed: int):
    return Pipeline([("scaler", "passthrough"), ("clf", HistGradientBoostingClassifier(random_state=seed, **params))])


def build_mlp(params: dict, seed: int):
    return Pipeline([("scaler", StandardScaler()), ("clf", MLPClassifier(random_state=seed, **params))])


LOGREG = ModelSpec("logistic_regression", "Linear (logistic regression)", build_logreg,
                   dict(class_weight="balanced", max_iter=1000))
KNN = ModelSpec("knn_distance", "Instance-based (distance-weighted KNN)", build_knn,
                dict(n_neighbors=15, weights="distance"))
RANDOM_FOREST = ModelSpec("random_forest", "Tree ensemble (bagging)", build_rf,
                          dict(n_estimators=100, max_depth=12, class_weight="balanced_subsample"))
HGB = ModelSpec("hist_gradient_boosting", "Boosting (histogram gradient boosting)", build_hgb,
                dict(learning_rate=0.05, max_iter=150, class_weight="balanced"))
MLP = ModelSpec("mlp", "Neural network (shallow MLP)", build_mlp,
                dict(hidden_layer_sizes=(64, 32), activation="relu", max_iter=100, early_stopping=True, validation_fraction=0.1))

from experiments.train_model_svm import build_linear_svm  # noqa: E402  (Person 3's model)

LINEAR_SVM = ModelSpec("linear_svm", "Kernel-based (linear SVM)", build_linear_svm,
                       dict(C=1.0, class_weight="balanced", max_iter=5000))

TEAMMATE_MODELS = [LOGREG, KNN, RANDOM_FOREST, HGB, MLP, LINEAR_SVM]
REGISTRY = {s.name: s for s in (XGBOOST, GAUSSIAN_NB, *TEAMMATE_MODELS)}
