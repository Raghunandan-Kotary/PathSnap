"""
Model 3 (Person 2): Random Forest Ensemble Classifier.
Algorithm Family: Tree Ensemble
Trains using shared evaluation harness on 80 training trips (5-fold group CV)
and evaluates on 10 student-test trips.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold

from experiments.metrics import compute_classification_metrics, compute_observation_top1_accuracy
from models.loader import FEATURE_COLUMNS

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "experiments" / "results"

def create_model_3(n_estimators: int = 100, max_depth: int = 12):
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )

def train_and_evaluate():
    print("=" * 80)
    print("Training Model 3: Random Forest (Tree Ensemble Family)")
    print("=" * 80)

    train_path = PROCESSED_DIR / "candidates_features_train.parquet"
    test_path = PROCESSED_DIR / "candidates_features_test.parquet"

    print("Loading processed feature tables...")
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    # Subsample 200,000 candidate rows for Random Forest training efficiency
    print("Stratified sampling 200,000 training candidates for ensemble training...")
    pos = train_df[train_df["label"] == 1]
    neg = train_df[train_df["label"] == 0].sample(min(160000, len(train_df[train_df["label"] == 0])), random_state=42)
    sub_train = pd.concat([pos.sample(min(40000, len(pos)), random_state=42), neg]).sample(frac=1.0, random_state=42)

    gkf = GroupKFold(n_splits=5)
    X = sub_train[FEATURE_COLUMNS].values
    y = sub_train["label"].values
    groups = sub_train["trip_id"].values

    fold_f1s = []
    fold_top1s = []

    print("\nRunning 5-Fold GroupKFold CV...")
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        model = create_model_3(n_estimators=80, max_depth=12)
        model.fit(X[train_idx], y[train_idx])
        
        val_sub = sub_train.iloc[val_idx]
        val_probs = model.predict_proba(X[val_idx])[:, 1]
        val_preds = (val_probs >= 0.5).astype(int)
        
        f1 = compute_classification_metrics(y[val_idx], val_preds, val_probs)["macro_f1"]
        top1 = compute_observation_top1_accuracy(val_sub, val_probs)
        fold_f1s.append(f1)
        fold_top1s.append(top1)
        print(f"  Fold {fold}: Macro F1 = {f1:.4f}, Top-1 Acc = {top1*100:.2f}%")

    cv_macro_f1 = float(np.mean(fold_f1s))
    cv_top1 = float(np.mean(fold_top1s))
    print(f"\nCV Mean Macro F1: {cv_macro_f1:.4f} (+/- {np.std(fold_f1s):.4f})")
    print(f"CV Mean Top-1 Acc: {cv_top1*100:.2f}%")

    # Fit on full subsample and evaluate on student_test
    print("\nFitting final Random Forest model and evaluating on 10 student-test trips...")
    final_model = create_model_3(n_estimators=100, max_depth=12)
    final_model.fit(X, y)

    X_test = test_df[FEATURE_COLUMNS].values
    y_test = test_df["label"].values

    test_probs = final_model.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= 0.5).astype(int)
    class_metrics = compute_classification_metrics(y_test, test_preds, test_probs)
    test_top1 = compute_observation_top1_accuracy(test_df, test_probs)

    print(f"  Student-Test Top-1 Accuracy: {test_top1*100:.2f}%")
    print(f"  Student-Test Macro F1:       {class_metrics['macro_f1']:.4f}")
    print(f"  Student-Test Log Loss:       {class_metrics['log_loss']:.4f}")

    out_dict = {
        "model_name": "Model 3: Random Forest",
        "family": "Tree Ensemble",
        "hyperparameters": {"n_estimators": 100, "max_depth": 12, "class_weight": "balanced_subsample"},
        "cv_macro_f1": cv_macro_f1,
        "cv_top1_accuracy": cv_top1,
        "student_test_macro_f1": class_metrics["macro_f1"],
        "student_test_top1_accuracy": test_top1,
        "student_test_log_loss": class_metrics["log_loss"]
    }

    out_file = RESULTS_DIR / "model3_rf_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out_dict, f, indent=2)
    print(f"\nSaved Model 3 results to {out_file}")

    return out_dict

if __name__ == "__main__":
    train_and_evaluate()
